"""
Entry point for the predictive pipeline.

Run with:
    python main.py

What it does, week 4:
    load config -> load data -> clean (fit-free, row-preserving) -> drop duplicates (training
    only) -> features/target -> lock the test set aside -> cross-validate [preprocessor + model]
    as one Pipeline on the development set -> report per fold and out-of-fold -> refit on the
    whole development set -> save results

The locked test set is set aside and never scored. Cross-validation fits and throws away one
model per fold: it estimates how good the *recipe* is. The model you would actually deploy is
the same pipeline refit on all development rows, which is what happens at the end.
"""
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    build_preprocessor,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
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
    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )

    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv,
        scoring=cv_config.get("scoring", "accuracy"),
        n_jobs=cv_config.get("n_jobs", 1),
    )

    report = cv_report(fold_scores, scoring=cv_config.get("scoring", "accuracy"))
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # CV estimates the recipe; the model you would deploy is this one
    pipeline.fit(X_dev, y_dev)
    print(f"Final model refit on all {len(X_dev)} development rows "
          f"({len(X_test)} test rows still locked away, never scored).")

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
