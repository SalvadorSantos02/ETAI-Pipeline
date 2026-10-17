"""
Entry point for the predictive pipeline.

Run with:
    python main.py

What it does, week 5:
    load config -> load data -> clean (fit-free, row-preserving) -> drop duplicates (training
    only) -> features/target -> lock the test set aside -> cross-validate [preprocessor + model]
    as one Pipeline on the development set -> [if tuning is enabled: nested CV of the tuning
    procedure, then tuning on all development rows] -> report per fold and out-of-fold -> refit
    on the whole development set -> save results

The locked test set is set aside and never scored. Cross-validation fits and throws away one
model per fold: it estimates how good the *recipe* is. The model you would actually deploy is
the same pipeline refit on all development rows, which is what happens at the end.
"""
import optuna
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
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

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

    # preprocessor + model as one object, so every fold re-learns its own medians,
    # category statistics and scale from its training rows alone
    pipeline = build_pipeline(config["preprocessing"], config["model"])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring=scoring, n_jobs=n_jobs)

    header = f"Hyperparameters from config.yaml: {config['model'].get('params')}"
    print(header)
    report = header + "\n" + cv_report(fold_scores, scoring=scoring)

    # week 5: hyperparameter tuning -- on the development set only, every candidate scored by
    # CV of the whole pipeline (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # 1. the honest estimate, on the same outer folds (its out-of-fold predictions replace
        #    the untuned ones in the reports below)
        print(f"\nNested cross-validation: {cv.get_n_splits()} outer folds x {n_trials} trials x "
              f"{inner_cv.get_n_splits()} inner folds ...")
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\n\nNested cross-validation (the tuning procedure, estimated honestly):\n"
        report += cv_report(nested_scores, scoring)

        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        print()
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # CV estimates the recipe; the model you would deploy is this one (tuned, if tuning is on)
    pipeline.fit(X_dev, y_dev)
    refit = (f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows "
             f"({len(X_test)} test rows still locked away, never scored).")
    if tuning_enabled:
        refit += f" Tuned hyperparameters: {study.best_params}"
    print(refit)
    report += "\n" + refit + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
