"""
Data diagnostics -- week 3.

The three techniques from the EDA notebook (`01_eda_introduction.ipynb`), lifted out of the
notebook and made generic: nothing here knows COMPAS's column names. What to test, which rules
to apply and which column is the identifier all come from the `diagnostics` section of
config.yaml.

    1. test_missingness_mechanism  -- is a column's missingness MCAR, or tied to something we observe?
    2. flag_invalid_values         -- values that are present but impossible, caught by domain rules
    3. find_duplicates             -- the same row twice, checked two ways

These are diagnostics: they report, and (for the invalid values) convert what they catch to NaN.
Deciding what to do about the findings is preprocessing's job, not theirs.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def _cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V effect size for a chi-square test of association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1)))


def verdict_from_max_v(max_v: float) -> str:
    """Turn the strongest association found into a verdict, so the threshold is code, not a judgment call."""
    if max_v >= 0.2:
        return "MAR / MNAR (possible pattern)"
    elif max_v >= 0.1:
        return "borderline -- worth a closer look"
    else:
        return "MCAR (scattered, no pattern found)"


def test_missingness_mechanism(
    df: pd.DataFrame,
    target_col: str,
    candidate_predictors: list,
) -> pd.DataFrame:
    """
    For `target_col`'s missing-value indicator, test association against each
    column in `candidate_predictors` via chi-square + Cramer's V.
    Returns one row per predictor, sorted by association strength (strongest first).

    The model's own target is deliberately not among the predictors: letting the label steer
    how the data is cleaned is a leak, one step earlier than fitting a model to it.
    """
    indicator = df[target_col].isna()
    rows = []
    for predictor in candidate_predictors:
        if predictor == target_col:
            continue
        sub = pd.DataFrame({"missing": indicator, "predictor": df[predictor]}).dropna(subset=["predictor"])
        if sub["predictor"].nunique() < 2 or sub["missing"].nunique() < 2:
            continue
        table = pd.crosstab(sub["missing"], sub["predictor"])
        chi2, p, _, _ = chi2_contingency(table)
        v = _cramers_v(table)
        rows.append({"predictor": predictor, "cramers_v": round(v, 3), "p_value": p, "n": len(sub)})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def missingness_verdicts(
    df: pd.DataFrame,
    candidate_predictors: list,
    columns: list = None,
) -> pd.DataFrame:
    """
    Run `test_missingness_mechanism` over every column that has missing values (or over
    `columns`, if given) and collapse each result into one verdict row.
    """
    if columns is None:
        columns = [c for c in df.columns if df[c].isna().any()]

    rows = []
    for col in columns:
        preds = [p for p in candidate_predictors if p != col]
        result = test_missingness_mechanism(df, col, preds)
        max_v = result.iloc[0]["cramers_v"] if len(result) else 0.0
        rows.append({
            "column": col,
            "n_missing": int(df[col].isna().sum()),
            "strongest_predictor": result.iloc[0]["predictor"] if len(result) else None,
            "max_cramers_v": max_v if len(result) else None,
            "verdict": verdict_from_max_v(max_v),
        })
    return pd.DataFrame(rows)


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> tuple:
    """
    Apply a dict of `{column: predicate}` domain rules and convert violations to NaN.

    Each predicate is a pandas expression over the column, exactly as written in config.yaml
    (for example `"18 <= age <= 100"`). A value that breaks its rule is present but impossible --
    an age of -3, a decile score of 23 on a 1-10 scale -- and "impossible but not missing" is
    still missing, so it is set to NaN before anything downstream treats it as real data.

    Returns `(df, report)`: a copy with violations converted, and one report row per rule.
    """
    df = df.copy()
    rows = []

    for column, rule in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = pd.DataFrame({column: numeric}).eval(rule)
        # only values that are actually present can violate a rule; already-missing stays missing
        violations = numeric.notna() & ~valid
        rows.append({
            "column": column,
            "rule": rule,
            "violations": int(violations.sum()),
            "examples": sorted(numeric[violations].unique().tolist())[:6],
        })
        df.loc[violations, column] = np.nan

    return df, pd.DataFrame(rows)


def find_duplicates(df: pd.DataFrame, id_column: str) -> dict:
    """
    Check for duplicate rows two ways, because they catch different failures.

    `.duplicated()` only finds rows identical in every column. The more common real-world
    failure is the same case entered twice under one identifier with a single field typed
    differently -- invisible to `.duplicated()`, caught by the id check. Both are reported;
    they agreeing on one dataset does not mean they will on the next.
    """
    exact_mask = df.duplicated()
    id_mask = df[id_column].duplicated() if id_column in df.columns else pd.Series(False, index=df.index)

    return {
        "exact_duplicate_rows": int(exact_mask.sum()),
        "repeated_ids": int(id_mask.sum()),
        "checks_agree": bool(exact_mask.sum() == id_mask.sum()),
        "duplicate_ids": df.loc[df[id_column].duplicated(keep=False), id_column].unique().tolist()
                         if id_column in df.columns else [],
    }
