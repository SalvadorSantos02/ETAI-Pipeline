"""
Entry point for the predictive pipeline.

Run with:
    python main.py                      # COMPAS (config.yaml)
    python main.py config_adult.yaml    # week 9: any other problem, through its own config

What it does, week 8 (week 9: plus predictions for a new, unlabelled file, when the config names one):
    load config -> load data -> clean (fit-free, row-preserving) -> drop duplicates (training
    only) -> features/target -> lock the test set aside -> cross-validate [preprocessor +
    feature selection (if enabled) + model] as one Pipeline on the development set, on every metric in cv.metrics, plus the no-model
    floor (a dummy) on the same folds -> [if tuning is enabled: nested CV of the tuning
    procedure, then tuning on all development rows, both on cv.selection_metric] -> out-of-fold
    reports (from probabilities) -> refit on the whole development set -> the features the
    final model keeps -> one row on the scoreboard (results/runs.csv) + the full report

The locked test set is set aside and never scored. Cross-validation fits and throws away one
model per fold: it estimates how good the *recipe* is. The model you would actually deploy is
the same pipeline refit on all development rows, which is what happens at the end.
"""
import os
import shutil
import sys
import tempfile
from datetime import datetime

import optuna
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.model import build_pipeline
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    print(f"Config: {config_path}")
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    df = load_data(config["data"]["path"])

    # fit-free and row-preserving: nothing here is learned from the data
    df = clean_dataset(df, config["diagnostics"])
    # training-only: the same person must not be counted twice, or land in both sets
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))

    X, y, extras = split_features_target(
        df, config["data"], config["preprocessing"]["mnar_indicator_sources"]
    )

    # the locked test set: carved out once, never scored, never looked at
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    # week 7: every metric is reported; the selection metric is what tuning maximises and
    # runs are ranked by. The scorers are built once and passed everywhere.
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV")
    print(f"Data:  {len(X_dev)} development rows   |   {len(X_test)} test rows locked away (not used)")

    # week 8: feature selection is a learned step (it looks at the labels), so it lives inside
    # the Pipeline and is re-fit on the training rows of every fold
    selection_config = config.get("feature_selection") or {}
    selection_enabled = selection_config.get("enabled", False)
    selection_label = f"vote, min_votes={selection_config.get('min_votes', 2)}" if selection_enabled else "off"
    print(f"Feature selection: {selection_label}")

    # a new, empty cache folder for this run only: the fitted prep and select steps are stored
    # there and reused when the same step is fit on the same rows again (a later tuning trial on
    # the same fold), so tuning the model doesn't refit them every time. Deleted at the end.
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")

    # preprocessor + selection + model as one object, so every fold re-learns its own medians,
    # category statistics, scale and kept features from its training rows alone
    pipeline = build_pipeline(config["preprocessing"], config["model"], selection_config, memory=cache_dir)
    # the no-model floor, on the same folds: what every model must beat
    floor_pipeline = build_pipeline(config["preprocessing"], {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv,
                                              {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()

    print(f"\nDefault hyperparameters ({config_path})")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from {config_path}: {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores

    # week 5: hyperparameter tuning -- on the development set only, every candidate scored by
    # CV of the whole pipeline (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but {config_path} has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = dict(search_spaces[model_type])
        # week 8: how many votes a feature needs is a hyperparameter like any other, tuned
        # together with the model (and so covered by the nested CV)
        if selection_enabled:
            search_space.update(selection_config.get("search_space") or {})
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # 1. the honest estimate, on the same outer folds (its out-of-fold probabilities replace
        #    the untuned ones in the reports below)
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (the tuning procedure, estimated honestly)")

        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<20s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<20s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=config["data"]["sensitive_attr"],
                                     reference_score=config["data"].get("reference_score"))

    # CV estimates the recipe; the model you would deploy is this one (tuned, if tuning is on)
    pipeline.fit(X_dev, y_dev)
    refit = (f"Final model: {model_type} refit on all {len(X_dev)} development rows "
             f"({len(X_test)} test rows still locked away, never scored).")
    report += "\n" + refit + "\n"

    # week 8: what the final model actually sees (after preprocessing, and after selection if on)
    kept = list(pipeline[:-1].get_feature_names_out())
    if selection_enabled:
        votes = dict(zip(pipeline.named_steps["prep"].get_feature_names_out(),
                         pipeline.named_steps["select"].votes_))
        kept_line = (f"Features kept ({len(kept)} of {len(votes)}): {', '.join(kept)}\n"
                     f"Votes (of 3): {', '.join(f'{name} {int(v)}' for name, v in votes.items())}")
    else:
        kept_line = f"Features kept (no selection, all {len(kept)}): {', '.join(kept)}"
    report += kept_line + "\n"

    # week 9: predictions for new, unlabelled rows (e.g. adult_new.csv), from the refit pipeline.
    # Same cleaning as the training file, but no drop_duplicate_rows(): every row needs a prediction.
    predict_config = config.get("predict") or {}
    predictions_line = ""
    if predict_config.get("path"):
        new_rows = load_data(predict_config["path"])
        new_clean = clean_dataset(new_rows, config["diagnostics"])
        X_new, _, _ = split_features_target(new_clean, config["data"],
                                            config["preprocessing"]["mnar_indicator_sources"])
        X_new = X_new[X_dev.columns]                     # the training columns, in the same order
        new_proba = pipeline.predict_proba(X_new)[:, 1]
        threshold = predict_config.get("threshold", 0.5)
        id_column = config["diagnostics"].get("id_column")
        predictions = pd.DataFrame({
            id_column: new_clean[id_column].values,
            "probability": new_proba,
            "prediction": (new_proba >= threshold).astype(int),
        })
        if len(predictions) != len(new_rows):
            raise RuntimeError(f"{len(new_rows)} rows in, {len(predictions)} predictions out")
        predictions_path = predict_config["output"]
        os.makedirs(os.path.dirname(predictions_path) or ".", exist_ok=True)
        predictions.to_csv(predictions_path, index=False)
        predictions_line = (f"Predictions: {len(predictions)} rows of {predict_config['path']} -> {predictions_path} "
                            f"(threshold {threshold}; {predictions['prediction'].mean():.1%} predicted 1)")
        report += predictions_line + "\n"

    # week 7: the scoreboard -- one row per run, with the floor and the untuned score on the
    # same folds, and the score on each fold (so two runs can be compared fold by fold)
    row = {"run": run, "data": config["data"]["path"], "model": model_type, "tuned": tuning_enabled,
           "feature_selection": selection_label,
           "encoder": config["preprocessing"]["encoder"], "scaler": config["preprocessing"]["scaler"],
           "params": str(config["model"].get("params")), "tuned_params": str(study.best_params) if tuning_enabled else "",
           "metric": selection_metric, "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)].copy()
    same["feature_selection"] = same["feature_selection"].fillna("off")   # runs before week 8
    board = (same.sort_values("mean", ascending=False)
             [["run", "model", "tuned", "feature_selection", "encoder", "scaler", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report)
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")
    print(kept_line)
    if predictions_line:
        print(predictions_line)

    # the cached prep/select fits are only valid for this run
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    # week 9: the config is the problem; config.yaml (COMPAS) when none is given
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
