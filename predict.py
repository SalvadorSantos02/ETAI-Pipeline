"""
Week 10: predict new people with the model saved by main.py -- no training.

Run with (from the project folder):
    python main.py                                                  # trains, saves models/model.joblib
    python predict.py data/compas_new_people.csv predictions.csv    # loads it, predicts every row

The input is a CSV with the same columns as the training data; the target is not needed. It goes
through the same fit-free cleaning as the training file (clean_dataset) but no duplicate dropping:
one row out per row in. Everything that was learned (medians or imputer models, encodings, scale,
kept features, the model) comes from the saved pipeline, so it is exactly what was fit in main.py.
"""
import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Returns the id, the probability of the positive class and the prediction (at 0.5) for every row."""
    clean = clean_dataset(df, config["diagnostics"])
    X, _, _ = split_features_target(clean, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    probability = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({"id": df[config["diagnostics"]["id_column"]],
                         "probability": probability,
                         "prediction": (probability >= 0.5).astype(int)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    # week 9's main.py takes the config as an argument; the same here, so the Adult model works too
    parser.add_argument("--config", default="config.yaml", help="the config main.py was run with")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config["output"]["model_path"])
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")
