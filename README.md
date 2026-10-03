# Salvador Santos -- ETAI Pipeline

Fork of [sofiacper/ETAI-Pipeline](https://github.com/sofiacper/ETAI-Pipeline), the baseline for
the Exploratory Topics in AI practical classes, grown week by week.

## Overview

The task: predict **two-year recidivism** (`two_year_recid`) -- will this person be rearrested
within two years? -- from the case facts available at screening time, using ProPublica's COMPAS
dataset. That is the data behind the 2016 investigation into a risk-assessment algorithm actually
used by courts in Broward County, Florida, to help inform bail and sentencing decisions.

What makes it interesting is the second question, the one ProPublica actually asked: is the model
equally accurate for everyone, or wrong more often, in one direction, for some groups? ProPublica
found a false-positive rate of 45% for Black defendants against 23% for white ones, hidden behind
similar overall accuracy. So `race` is **excluded from the model's inputs** and carried alongside
the data purely to audit fairness afterwards -- never to train on. Leaving it out does not make
the bias disappear, which is the point.

See `data/README.md` for the full problem description and data dictionary.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning rules + leak-safe ColumnTransformer + the locked test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # cross-validation, out-of-fold reports + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramer's V, domain-rule invalid-value detection, duplicate check done two ways) and a rewritten `src/preprocessing.py` (fit-free `clean_dataset()`, `_was_missing` indicators for the MNAR columns, a leak-safe `ColumnTransformer` built by `build_preprocessor()`, and the split) replace week 2's naive `dropna()` / `get_dummies()`. The encoder/scaler pair was chosen empirically over 15 repeated splits rather than by convention; three redundant columns dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections, and `main.py` now fits preprocessor and model as **one** `Pipeline`, so nothing is fitted outside the training rows |

## Preprocessing decisions

One line per column that needed a decision -- what was wrong, which mechanism, what was done, and
why. This is the EDA notebook's verdict table, in prose.

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 155 missing, plus 9 impossible values (`-3`, `5`) | MCAR -- strongest Cramer's V 0.038 | invalid values to `NaN`, then median impute. No indicator: there is no pattern to preserve |
| `juv_fel_count` | 222 missing, plus 5 negative counts | MCAR -- V 0.042 | negatives to `NaN`, then median impute, no indicator |
| `priors_count` | 426 missing (incl. `-` placeholders), 6 impossible values (`250`, `500`) | **MNAR** -- V 0.391 against `age_cat` | median impute **and** a `priors_count_was_missing` flag, so the model can still see the pattern the fill erases |
| `c_charge_degree` | 233 missing | **MNAR** -- V 0.349 against `age_cat` | mode impute **and** a `c_charge_degree_was_missing` flag |
| `race` | ~1% missing behind `?` / `n/a` placeholders | MCAR | mode impute, no indicator -- excluded from model features anyway |
| `sex` | 60 missing, incl. placeholders | MCAR -- V 0.025 | mode impute, no indicator |
| `decile_score` | 6 values off the documented 1-10 scale (`0`, `15`, `23`) | domain rule | to `NaN`; not a model input, kept only to compare against COMPAS |
| `sex`, `race`, `c_charge_degree`, `score_text` | the same category under several spellings (`AFRICAN-AMERICAN`, `african-american`, `African American`), plus `-` and `?` posing as categories | data entry | canonicalized to one spelling per category; placeholder tokens to `NaN` |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` -- both checks agreed | data entry | dropped, first occurrence kept (7286 -> 7214 rows) |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant (r = 1.00 with `priors_count` and `age`; `juvenile_total` is the sum of the three juvenile counts, caught by VIF) | multicollinearity | dropped |

**Encoder / scaler:** all 16 encoder x scaler combinations were scored with logistic regression
over 15 repeated 75/25 splits. **target encoding + standard scaling** won at 0.6693 mean
accuracy -- but a *paired* comparison against the runner-up (target + robust, also 0.6693) gives
a difference of +0.0000 with a standard error of 0.0002, well inside noise. Either would be
defensible; the grid picks one so the pipeline has a single answer, not because the gap decides
anything. Worth re-checking once the model stops being a scaling-sensitive linear one.

**Not done yet, on purpose:** threshold-independent metrics (ROC-AUC, PR-AUC) and a calibration
check. The grid search above also tunes preprocessing on the same splits it scores on -- a mild
leak, acknowledged in the notebook and left until cross-validation arrives in week 4.

## Model evaluation

Two decisions, both made once and written into `config.yaml`:

**A locked test set** -- 20% of the rows, stratified, seed 42, carved out by `split_dev_test()`
and never used to fit, tune, compare or choose anything. Changing that seed later would silently
hand me a test set I had already partly seen, which is why the config says not to.

**Stratified 5-fold cross-validation of the whole pipeline** on the remaining 5,771 development
rows. The preprocessor sits *inside* the pipeline, so every fold re-learns its own medians,
category statistics and scale from its own training rows -- the validation fold never influences
its own preprocessing. Same `cv.random_state` for every model, so the comparison is like-for-like.
Reports are computed on out-of-fold predictions: every row is scored by the one fold model that
did not train on it.

| Model | Holdout accuracy (W3) | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|
| Dummy (majority class) | — | 0.549 ± 0.000 | −0.000 |
| Logistic regression | 0.678 | 0.672 ± 0.013 | +0.003 |
| Decision tree (`max_depth=5`) | 0.665 | **0.675 ± 0.018** | +0.011 |
| Random forest (300 trees, untuned) | — | 0.650 ± 0.018 | +0.083 |

**Which number would I trust?** The CV one. A single holdout score is one draw: week 3's decision
tree scored 0.665 on one split and 0.675 across five, and the ±0.018 spread says that difference
is the split talking, not the model. The CV figure comes with an error bar, which is the whole
point.

**Does the week 2/3 conclusion still hold?** No, and it never had the evidence to. I concluded
week 2's logistic regression was best on 0.678 vs 0.665 -- a gap of 0.013, which is smaller than
a single standard deviation of either model's fold scores. Under CV, logistic regression (0.672)
and the decision tree (0.675) are indistinguishable; neither is "the best model", and picking one
on the old numbers would have been reading noise.

Two things the table makes visible that no previous week could. The **dummy** floor is 0.549 --
the share of the majority class -- so every real model is buying about 12 accuracy points over
guessing, which is modest and worth knowing. And the **random forest**, the most powerful model
here, is the *worst* at 0.650 with a +0.083 train-validation gap: 300 untuned trees memorise the
development set and generalise no better for it. The gap column is doing exactly the job it was
added for.

**Fairness, out-of-fold on the development set:** our model's false-positive rate is 0.36 for
African-American defendants against 0.23 for Caucasian ones (COMPAS's own score: 0.45 against
0.23). Same direction as ProPublica's finding, on 5,771 rows instead of 1,443, with `race` never
a feature.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, cleans the data, sets the locked test set aside, cross-validates the
pipeline on the development set, and prints:
- the **per-fold table** -- train and validation score for each of the 5 folds, plus the gap
  between them, and the mean +/- std of each column. A large, consistent gap is overfitting.
- a **classification report on the out-of-fold predictions**, so every development row is scored
  by a model that never trained on it
- a **false-positive-rate-by-race comparison** between our model and COMPAS's own score, on those
  same out-of-fold predictions

It then refits the pipeline on the whole development set -- cross-validation estimates how good
the *recipe* is and throws its five models away; this last fit is the model you would actually
use. The locked test set is never scored.

All of this is also saved to a timestamped file in `results/` (e.g. `results/run_20261003_191844.txt`),
so it doesn't just scroll past in your terminal -- open it later, or change something in
`config.yaml` (like the model type) and compare the new file to the last one. `results/` is
created automatically the first time you run the pipeline, and isn't tracked in git (see
`.gitignore`) since it's generated output, not source.

To reproduce the model comparison table above, switch `model.type` in `config.yaml` between
`dummy`, `logistic_regression`, `decision_tree` and `random_forest` (the matching `params` are
commented there) and run `python main.py` for each.

### A note on scikit-learn versions

`02_preprocessing.ipynb` builds the target encoder as
`TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed))`,
which needs scikit-learn >= 1.9. The newest release currently installable here is 1.7.2, where
`TargetEncoder` takes `cv`, `shuffle` and `random_state` directly and builds the stratified folds
itself for a binary target. `src/preprocessing.py` uses that form -- same cross-fitting, same
seed, so a row's own label never reaches its own encoding. Worth reverting to the notebook's
one-liner once 1.9 is available.

## Dataset

See `data/README.md`.
