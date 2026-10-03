"""
Preprocessing -- week 4: a leak-safe recipe, plus the locked test set.

A recipe is a set of *rules* ("fill `priors_count` with the median"). The *fitted values* (which
median? which categories? what scale?) are learned again from the training rows of every split --
which is what makes preprocessing leak-safe, and why cross-validation can put the whole recipe
inside every fold.

So the work splits in two:

  Rules, fit-free and target-agnostic -- `clean_dataset()`. Canonicalizing "MALE" to "Male" or
  turning an age of -3 into NaN learns nothing from the data, so it can run on the whole file at
  once, and on unlabeled data being scored for real.

  Fits -- everything inside `build_preprocessor()`. Imputed medians, encoder statistics, scaler
  centre and spread. This object is placed *inside* the model's sklearn Pipeline, so it is fit
  only on the training part of each fold.

`clean_dataset()` is row-preserving (week 4): every input row comes out. Dropping duplicates is a
training-only decision and lives in `drop_duplicate_rows()`, because at prediction time every row
needs a prediction.
"""
import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


# ---------------------------------------------------------------- rules (fit-free)

def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies a dict of {column: {"min": ..., "max": ...}} domain rules (either bound is
    optional) and converts violations to NaN **in place** on `df`. An "impossible but
    not missing" value (an age of -3, a COMPAS decile score of 15) counts as missing
    once this runs -- `.isna()` alone would never have caught it.

    Returns a small report: how many violations were found per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies the week 3 diagnosis: category cleanup, domain-rule/placeholder -> NaN
    conversion, and redundant-column removal. Target-agnostic -- safe to call on
    label-free inference data, since none of this depends on a target column.

    Row-preserving (week 4): every input row comes out, in the same order. Removing
    duplicate rows is a *training-only* decision and lives in `drop_duplicate_rows()` --
    at prediction time every row needs a prediction (a Kaggle submission needs one per id).
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    return out


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    TRAINING DATA ONLY (week 4). Drops exact duplicate rows and repeated ids (keeping
    the first), so the same person can't be counted twice -- or land in both the
    development and the locked test set. Must run *before* `split_dev_test()`.

    Never call this on data you're predicting for: every row there needs a prediction.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


# ---------------------------------------------------------------- features / target / split

def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR-diagnosed column, before that
    column gets imputed -- so a model can still see the pattern even though the fill
    value itself (median/mode) can't carry it. Target-agnostic."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `y` is `None` and `extras` has no target column when called
    on label-free inference data -- nothing downstream requires the target to be present.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Sets the final test set aside (week 4 -- replaces week 2/3's `split_train_test`).

    Stratified split of X, y and the extras frame (race/score_text, kept for the fairness
    report) together, so all three stay row-aligned. Returns a *development* set and a
    *locked test set*:
      - development set: everything we're allowed to learn from and compare models on.
        Cross-validation (src/evaluate.py) splits it again into train/validation folds.
      - locked test set: never used to fit, tune, compare or choose anything. Its size and
        seed live in config.yaml's `test_set` section and are never changed after today.
    """
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test


# ---------------------------------------------------------------- the fitted recipe

_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    # The notebook passes cv=StratifiedKFold(5, shuffle=True, random_state=seed), which needs
    # scikit-learn >= 1.9. The latest release available here is 1.7.2, where TargetEncoder takes
    # the same thing as cv / shuffle / random_state and builds the stratified folds itself for a
    # binary target. Same cross-fitting, same seed -- a row's own label never reaches its own
    # encoding. Revert to the notebook's one-liner once 1.9 is installable.
    "target": lambda seed: TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Factory: builds a leak-safe ColumnTransformer for the chosen encoder/scaler pair --
    read from `config.yaml`'s `preprocessing` section (chosen there, not hardcoded
    here). Every encoder tolerates unseen categories at transform time. Nothing is fit
    here: fitting happens later, on the training part of each CV fold only, because this
    object is placed *inside* the model's sklearn Pipeline (see main.py).
    """
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[encoder_name](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])

    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])
