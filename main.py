"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean (fit-free) -> features/target -> split
    -> fit [preprocessor + model] as one Pipeline -> evaluate (train & test) -> save results

Week 3: preprocessing is no longer a single naive step. The fit-free cleaning runs once on the
whole file; everything that is *fitted* (imputer, encoder, scaler) lives inside the Pipeline, so
it is fit on the training rows only and applied unchanged to the test rows.
"""
import json

import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    build_preprocessor,
    clean_dataset,
    split_features_target,
    split_train_test,
)
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])

    with open(config["data"]["diagnosis_path"], "r") as f:
        diagnosis = json.load(f)

    # fit-free: canonicalize categories, impossible values -> NaN, de-dup, drop redundant columns
    df = clean_dataset(df, diagnosis)

    X, y, extras = split_features_target(df)
    X_train, X_test, y_train, y_test, extras_test = split_train_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # preprocessor + model as one object: fit() learns the medians, categories and scale from the
    # training rows alone, and predict() applies exactly those to rows it has never seen
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(
            encoder_name=config["preprocessing"]["encoder"],
            scaler_name=config["preprocessing"]["scaler"],
            imputation=config["preprocessing"].get("imputation"),
        )),
        ("model", build_model(config["model"])),
    ])
    pipeline.fit(X_train, y_train)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
