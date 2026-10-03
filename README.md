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
│   ├── data_diagnostics.py # week 3: missingness mechanism, domain rules, duplicate checks
│   ├── preprocessing.py    # cleaning recipe + leak-safe ColumnTransformer + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── diagnosis_log.json   # the EDA notebook's findings, read by clean_dataset()
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

## Best Model

| Week | Model | Preprocessing | Train acc | Test acc | Gap |
|---|---|---|---|---|---|
| 2 | logistic regression (`max_iter=1000`) | naive: `dropna()` + one-hot | 0.679 | **0.678** | +0.001 |
| 2 | decision tree (`max_depth=5`) | naive | 0.680 | 0.668 | +0.012 |
| 3 | decision tree (`max_depth=5`) | week 3 recipe: target encoding + standard scaling | 0.684 | 0.665 | +0.020 |

**Current best: week 2's logistic regression, at 0.678 test accuracy.** Worth being honest about
that rather than claiming the new pipeline won: better preprocessing did *not* buy accuracy here.

Two reasons it is still progress. First, the two numbers are not measured on the same rows --
week 2 dropped every row with a missing value, scoring on an easier, cleaner subset of 6,256
rows; week 3 keeps all 7,214 and has to predict the messy ones too. Second, the fairness
table went from 16 garbled race labels to 6 real categories, which is the difference between a
number and a number that means something: our model's false-positive rate is now a readable 0.28
for African-American defendants against 0.16 for Caucasian ones (COMPAS's own score: 0.44 against
0.24). Same direction as ProPublica's finding, with `race` never once a feature.

A fair comparison between the two needs cross-validation on a common footing -- week 4's job.

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

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
