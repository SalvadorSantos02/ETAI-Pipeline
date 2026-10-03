"""
Preprocessing -- week 3: from diagnosis to a deployable recipe.

Week 2's version did one naive thing (`dropna()` + `get_dummies()`) and one real one (holding
`race` out of the model's inputs). This week the file grows to own the whole "raw data in,
model-ready split out" job, built on what the EDA notebook diagnosed.

The one-line rule it is built around:

    From the split onward, the held-out rows exist to imitate the future -- every preprocessing
    decision may look only at the training rows.

So the work is split in two. What is *target-agnostic and fit-free* (canonicalizing "MALE" to
"Male", converting impossible values to NaN, dropping duplicates and redundant columns) happens
once, up front, on the whole file: none of it learns anything that could leak. What is *fitted*
(an imputed median, an encoder's category list, a scaler's mean) lives inside a ColumnTransformer
that is fit on the training rows only and applied unchanged to everything else.

Two habits, established now rather than retrofitted later: every encoder tolerates a category it
has never seen at transform time, and nothing here requires the target column to be present --
the same code has to run on unlabeled data being scored for real.
"""
import numpy as np
import pandas as pd
from category_encoders import CountEncoder, TargetEncoder
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
)

TARGET = "two_year_recid"
SENSITIVE_ATTR = "race"
COMPAS_OWN_SCORE = ["decile_score", "score_text"]
DROP_ALWAYS = ["id", TARGET, SENSITIVE_ATTR] + COMPAS_OWN_SCORE

NUMERIC_FEATURES = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
CATEGORICAL_FEATURES = ["sex", "age_cat", "c_charge_degree"]
MNAR_INDICATOR_SOURCES = ["priors_count", "c_charge_degree"]


# ---------------------------------------------------------------- cleaning (fit-free)

def find_placeholder_rows(series: pd.Series, tokens: set) -> pd.Series:
    return series.astype(str).str.strip().isin(tokens)


def canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnosis: dict) -> pd.DataFrame:
    """
    Apply the EDA notebook's diagnosis: category cleanup, domain-rule / placeholder ->
    NaN conversion, de-duplication, redundant-column removal. Target-column-agnostic --
    safe to call on label-free inference data.
    """
    out = df.copy()
    placeholder_tokens = set(diagnosis["placeholder_tokens"])

    # numeric columns that loaded as text because of placeholder tokens
    for col in ["priors_count", "prior_offenses"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    # domain-rule violations -> NaN
    out.loc[out["age"].notna() & ~out["age"].between(18, 100), "age"] = np.nan
    out.loc[~out["decile_score"].between(1, 10), "decile_score"] = np.nan
    out.loc[out["juv_fel_count"].notna() & (out["juv_fel_count"] < 0), "juv_fel_count"] = np.nan
    out.loc[out["priors_count"].notna() & ((out["priors_count"] < 0) | (out["priors_count"] > 60)), "priors_count"] = np.nan

    # category canonicalization (also folds placeholder tokens in race/sex/c_charge_degree/score_text to NaN)
    out = canonicalize_categories(out, diagnosis["canonical_maps"], placeholder_tokens)

    # duplicates: exact row dupes and repeated ids point at the same rows here -- drop, keep first
    out = out.drop_duplicates()
    if "id" in out.columns:
        out = out.drop_duplicates(subset="id", keep="first")

    # redundant columns found via multicollinearity
    cols_to_drop = [c for c in diagnosis["columns_to_drop"] if c in out.columns and c != "id"]
    out = out.drop(columns=cols_to_drop)

    return out


# ---------------------------------------------------------------- features / target

def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR column, BEFORE that column gets imputed.
    Target-agnostic -- safe on label-free inference data."""
    out = df.copy()
    for col in MNAR_INDICATOR_SOURCES:
        out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame):
    """Splits into (X, y, extras). `y` is None and `extras` has no `two_year_recid`
    column when called on label-free inference data -- nothing downstream requires the target."""
    df = add_missingness_indicators(df)
    y = df[TARGET] if TARGET in df.columns else None
    extras_cols = [c for c in [SENSITIVE_ATTR, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None
    feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]
    X = df[feature_cols]
    return X, y, extras


# ---------------------------------------------------------------- the fitted recipe

def build_preprocessor(encoder_name: str, scaler_name: str, imputation: dict = None) -> ColumnTransformer:
    """Factory: builds a leak-safe ColumnTransformer for a given encoder/scaler choice.
    Every encoder tolerates unseen categories at transform time (handle_unknown='ignore'
    or its category_encoders equivalent) -- fit on train, applied unchanged to test/inference.

    `imputation` is the plan from config.yaml -- `{column: {strategy, indicator}}`. Only the
    strategies are read here; the `indicator` flags are what `MNAR_INDICATOR_SOURCES` already
    encodes, and they are applied before this runs. Falls back to median / most_frequent.
    """
    imputation = imputation or {}

    def strategy_for(columns: list, default: str) -> str:
        chosen = {imputation[c]["strategy"] for c in columns if c in imputation}
        # one imputer per block, so a block with mixed strategies keeps the default
        return chosen.pop() if len(chosen) == 1 else default

    scalers = {
        "none": "passthrough",
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    encoders = {
        "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "count": CountEncoder(handle_unknown=0, handle_missing=0),
        "target": TargetEncoder(handle_unknown="value", handle_missing="value"),
    }

    numeric_indicator_cols = [f"{c}_was_missing" for c in MNAR_INDICATOR_SOURCES]
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=strategy_for(NUMERIC_FEATURES, "median"))),
        ("scale", scalers[scaler_name]),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=strategy_for(CATEGORICAL_FEATURES, "most_frequent"))),
        ("encode", encoders[encoder_name]),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("indicators", "passthrough", numeric_indicator_cols),
    ])


# ---------------------------------------------------------------- the split (week 2's job)

def split_train_test(X, y, extras, test_size: float, random_state: int):
    """
    The stratified train/test split -- week 2's original job, still here.

    `extras` (race and COMPAS's own score) is split alongside X and y so the fairness check can
    line the test rows up with their group afterwards. It is never a model input.
    """
    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, extras_test
