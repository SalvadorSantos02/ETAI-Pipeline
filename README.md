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
├── main.py                # entry point: python main.py [config file]; saves the final model
├── predict.py             # week 10: python predict.py <input.csv> [output.csv] -- predicts with the saved model
├── config.yaml             # all tunable settings live here (COMPAS, the default)
├── config_adult.yaml       # week 9: the same pipeline on the Adult income data
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning rules + leak-safe ColumnTransformer + the locked test split
│   ├── model.py            # model registry (incl. oblique forest, AdaBoost, the boosters, XGBoost, Naive Bayes) + VoteSelector + build_pipeline() (preprocessing + selection + model)
│   ├── evaluate.py         # scorers, cross-validation on several metrics, out-of-fold reports, fairness check, paired test
│   ├── tuning.py           # Optuna tuning, nested cross-validation, tuning report
│   └── results.py          # the scoreboard (results/runs.csv) + each run's report on disk
├── results/                # created automatically -- one file per run (not tracked in git); results/adult/ for Adult
├── models/                 # week 10: model.joblib, the final fitted pipeline (created by main.py, not tracked in git)
├── predictions/
│   └── adult_new_predictions.csv   # week 9: the final Adult model's predictions for adult_new.csv
└── data/
    ├── compas_two_year_recidivism.csv
    ├── compas_new_people.csv   # week 10: 20 new people, no target, for predict.py
    ├── README.md            # problem description + full data dictionary
    ├── adult_train.csv      # week 9: 6,000 people, with the target
    ├── adult_new.csv        # week 9: 2,000 new people, no target
    └── ADULT_README.md      # week 9: Adult's columns
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramer's V, domain-rule invalid-value detection, duplicate check done two ways) and a rewritten `src/preprocessing.py` (fit-free `clean_dataset()`, `_was_missing` indicators for the MNAR columns, a leak-safe `ColumnTransformer` built by `build_preprocessor()`, and the split) replace week 2's naive `dropna()` / `get_dummies()`. The encoder/scaler pair was chosen empirically over 15 repeated splits rather than by convention; three redundant columns dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections, and `main.py` now fits preprocessor and model as **one** `Pipeline`, so nothing is fitted outside the training rows |
| 4 | Preprocessing review + cross-validation | A locked 20% test set (`split_dev_test()`, stratified, seed 42, never scored) and stratified 5-fold CV of the whole pipeline on the development set (`src/evaluate.py`): per-fold train/validation/gap, out-of-fold classification and fairness reports. `clean_dataset()` became row-preserving, de-duplication moved to a training-only `drop_duplicate_rows()`, and `src/data_diagnostics.py` was folded into preprocessing; random forest added as a model |
| 5 | CV & hyperparameter tuning | `src/tuning.py` (new): `tune_pipeline()` runs a seeded Optuna TPE study where every trial cross-validates the whole pipeline on the same folds; `nested_cross_validate()` gives the honest estimate of "tune, then refit"; `tuning_report()` prints the top trials and the optimism check. `build_pipeline()` in `src/model.py` is now the one place the estimator is assembled. `config.yaml` gains a `tuning` section (on/off, trials, folds, seed, one search space per model) and the model is the decision tree again; `optuna` added to requirements |
| 6 | Ensembles in depth -- random forest, oblique random forest, AdaBoost | `src/model.py` gains two model builders, registered in `_MODELS` next to the classes: `oblique_random_forest()` (a `BaggingClassifier` of `SparseRandomProjection` -> `DecisionTreeClassifier` pipelines, so every split is a random hyperplane, a new set of directions per tree) and `adaboost()` (`AdaBoostClassifier` over trees of a chosen depth). `config.yaml`: the random forest is the model `main.py` runs (200 trees, fixed), and `tuning.search_spaces` has one space per ensemble -- tree size and `max_features` for the forest (no longer `n_estimators`), the same tree size through `model__estimator__tree__...` plus `n_components` for the oblique forest, base-tree depth, rounds and learning rate for AdaBoost. No other file and no package changed |
| 7 | Boosting & metric evaluation | `src/model.py`: `gradient_boosting`, `hist_gradient_boosting` and `naive_bayes` (`GaussianNB`, a fast probabilistic baseline) in `_MODELS`, scikit-learn's classes as they are. `src/evaluate.py`: `build_scorers()` (scikit-learn's names plus `fbeta_<beta>`); `cross_validate_pipeline()` scores every metric and returns out-of-fold **probabilities**; the classification report (now with a confusion matrix) and the fairness check take those probabilities and a threshold; `corrected_paired_test()` (Nadeau & Bengio). `src/tuning.py`: tuning maximises the selection metric's scorer, and nested CV scores every metric on each outer fold. `src/results.py`: `save_scores()` appends every run to the scoreboard. `main.py`: the scorers, the no-model floor on the same folds, the scoreboard row and board. `config.yaml`: `cv.metrics` (8) and `cv.selection_metric: roc_auc` replace `scoring`; gradient boosting is the model `main.py` runs; search spaces for the three new models; `class_weight: balanced` in the commented params that support it; `output.scoreboard`. No package changed |
| 8 | XGBoost & feature selection | `src/model.py`: `"xgboost": XGBClassifier` in `_MODELS`; a new `VoteSelector` (a random forest and gradient boosting vote for the features with above-average importance, an L1 logistic regression for those it doesn't shrink to zero; a feature is kept with at least `min_votes` of the 3 votes); `build_pipeline()` takes the `feature_selection` section and a cache folder (`memory`) and puts the selector between `prep` and `model`, so it is re-fit on the training rows of every fold. `main.py`: a fresh cache folder per run (`tempfile.mkdtemp`, removed with `shutil.rmtree` at the end) so tuning doesn't refit prep and selection in every trial; with tuning on, `select__min_votes` joins the model's search space (and so the nested CV); prints the features the final model keeps, with each one's votes; scoreboard column `feature_selection`. `src/results.py`: the report header records the selection setting. `config.yaml`: `model.type: xgboost` (300 trees, learning rate 0.05, depth 4), the `xgboost` search space with its in-split brakes (`min_child_weight`, `reg_lambda`, `reg_alpha`, `gamma`), and the `feature_selection` section (on, `min_votes: 2`, tuned 1-3). `requirements.txt`: `xgboost>=2.0` |
| 9 | Pipeline implementations -- beyond COMPAS (option A: new data, Adult income) | `main.py` takes the config file as an argument (`python main.py config_adult.yaml`; `config.yaml` when none is given) and, when the config has a `predict` section, predicts a new unlabelled file with the refit pipeline (same cleaning, no duplicate dropping, one row out per row in). `src/preprocessing.py`: `clean_dataset()` turns placeholder tokens into missing values in every text column, not only the ones with a canonical map; `split_features_target()` reads the reference score's column from the config instead of naming `score_text`. `src/evaluate.py`: `fairness_report()` audits an optional `reference_score` from the config instead of assuming COMPAS's. `config.yaml`: `data.reference_score` (COMPAS's `score_text`, as before). New `config_adult.yaml` (data, diagnostics, preprocessing, its own `output` folder, `predict`), the Adult data in `data/` and the predictions in `predictions/`. The COMPAS run gives exactly the same numbers as before (checked on the same folds) |
| 10 | Missing values & deployment | `src/preprocessing.py`: `_NUMERIC_IMPUTERS` (`median`, `mean`, `knn` = `KNNImputer`, `iterative` = `IterativeImputer`, i.e. MICE with one completed dataset, `random_forest` = `IterativeImputer` over a `RandomForestRegressor`, the missForest idea); `build_preprocessor()` now **scales the numeric columns first and imputes second**, so KNN's distances compare columns on one scale, and stops on an unknown strategy. None of the imputers sees the target, so all of them also run on new data. `main.py`: the imputer is printed and recorded on the scoreboard (new column `imputer`); after the final refit the pipeline is saved with `joblib` to `output.model_path`. New `predict.py`: loads the saved pipeline and predicts a CSV from the terminal, one row out per row in. `src/results.py`: the imputer in the report header. Both configs: `imputation` options (`numeric_strategy`, `n_neighbors`, `max_iter`, `n_estimators`) and `output.model_path` (Adult has its own file). `.gitignore`: `models/`, `predictions.csv`. `requirements.txt`: `joblib`. `data/compas_new_people.csv`. With `median`, the run gives exactly the same fold scores as before (scaling then taking the median is the same as the reverse) |

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

### Hyperparameter tuning

`src/tuning.py` tunes with **Optuna**: a TPE sampler seeded from `tuning.random_state`, 30 trials
(15 for the random forest, which is ~100x slower per trial), each trial scored by 5-fold stratified
CV. The search space lives in `config.yaml`, one per model type, keyed by Pipeline parameter names
(`model__max_depth`, `model__C`, ...). It doesn't leak, for three reasons: it only ever sees the
development set (the test set stays locked); every trial clones the **whole** pipeline, so the
imputer, encoder and scaler are re-fit inside every fold of every trial; and every trial uses the
same folds, so candidates are compared like-for-like. The best trial's score is still the maximum
of many noisy numbers, so it is never the reported one: **nested CV** (tuning re-run from scratch
inside each of the 5 outer folds, scored on the outer fold it never saw) gives the honest estimate,
and the hyperparameters kept come from one more tuning run on all development rows.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.610 ± 0.018 (fully grown) | 0.672 ± 0.017 | 0.679 | +0.002 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Logistic regression | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | `C=0.072` |
| Random forest | 0.650 ± 0.018 (300 trees) | **0.679 ± 0.013** | 0.683 | +0.002 | `n_estimators=128`, `max_depth=8`, `min_samples_leaf=36`, `max_features=0.5` |

Optimism = mean tuning score inside the outer folds (`inner_best`) minus the mean honest score on
them. Default means `random_state` only for the tree, `max_iter=2000` for LR, 300 trees for the
forest; week 4's tree had a hand-picked `max_depth=5`.

The two tree models gained the most, and their gains are well beyond the fold-to-fold noise: the
tree went from 0.610 to 0.672 (+0.062, more than three standard deviations), the forest from 0.650
to 0.679 (+0.029), and in both the train-validation gap collapsed from about +0.08 to +0.01 --
tuning bought them a brake on memorising, which is what their defaults lacked. Logistic regression
gained +0.001, which is nothing: it was never overfitting (gap +0.003), and `C` only adjusts a
model with ten coefficients that was already about as simple as the data allows. Tuning did
change the ranking on paper -- the forest moves from worst to first -- but the three tuned models
sit within 0.007 of each other, less than one fold standard deviation, so they are effectively
tied. For the forest I would report **0.679 ± 0.013 from nested CV**, not the 0.683 of its best
trial, because nested CV is the only one of the two that was scored on rows that took no part in
choosing it. The forest picked a deeper `max_depth` (8) than the lone tree (6): averaging many
decorrelated trees cancels part of each tree's variance, so each one can afford to grow further.

### Ensembles

Week 6 adds the oblique random forest and AdaBoost, and re-tunes the random forest with a new search
space (tree size and `max_features`; `n_estimators` fixed at 200 because more trees never overfit,
they only cost time). Every row comes from `python main.py` with that `model.type`, tuning on (30
Optuna trials x 5 inner folds, nested over the same 5 outer folds), on the same development set
and folds as every week since week 4.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | — |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Random forest | 0.650 ± 0.019 (+0.082) | 0.681 ± 0.015 (+0.009) | 0.683 | `max_depth=22`, `min_samples_leaf=42`, `max_features=0.5` |
| Oblique random forest | 0.647 ± 0.011 (+0.114) | 0.681 ± 0.013 (+0.012) | 0.683 | `max_depth=6`, `min_samples_leaf=3`, `n_components=5` |
| AdaBoost | 0.676 ± 0.012 (+0.006) | **0.682 ± 0.017** (+0.004) | 0.684 | base `max_depth=4`, `n_estimators=220`, `learning_rate=0.025` |

Defaults: `max_iter=2000` for LR, `random_state` only for the tree (fully grown), 200 fully grown
trees for both forests (`n_components=10` for the oblique one), 200 stumps at `learning_rate=1.0`
for AdaBoost. Run times with tuning on, on a laptop: about 15-20 minutes per ensemble (900 fits).

I would choose AdaBoost, but only just: its nested 0.682 is ahead of the random forest by +0.001
and of the oblique forest by +0.001 on average, with folds going both ways, so the three tuned
ensembles are tied. Against last week's models they gain about one point (+0.010 over both the
tuned tree and logistic regression), which is smaller than the fold-to-fold std of ~0.015, so it is
not yet a convincing lead. The chosen hyperparameters do match how each model works: AdaBoost picked
weak base trees (depth 4, against the fully grown trees the forests start from) and paired many
rounds with a small learning rate, the two knobs that trade off; the random forest kept its trees
deep (`max_depth=22`) but stopped them at leaves of ~40 people, so averaging handles the variance
while the leaf size keeps every tree from memorising the identical noisy rows all trees share; the
oblique forest got to the same place with shallower trees on 5 random directions. Tuning mattered
most where the defaults overfit -- both forests' gaps of +0.08 / +0.11 dropped to about +0.01 --
and least for AdaBoost with stumps, which was already well behaved. I would report **AdaBoost,
0.682 ± 0.017 from nested CV**, not the 0.684 of its best trial; that all five models end between
0.672 and 0.682 says the ceiling is the information in these 10 features, not the model.

### Scoreboard

Week 7 changes how every model is judged: each fold is scored on eight metrics (`cv.metrics`)
from out-of-fold probabilities, and one of them, the **selection metric**, is what tuning maximises
and runs are ranked by. Every row below is one `python main.py` (one per model, appended to
`results/runs.csv`), tuned where the model has a search space (30 Optuna trials x 5 inner folds,
nested over the same 5 outer folds as every week since week 4). `Untuned` is the same model with
its `config.yaml` params; the floor is the dummy on the same folds. The numbers are **ROC-AUC**, so
they don't compare with the accuracy tables above.

| Model | Untuned ROC-AUC (CV) | Tuned ROC-AUC (nested CV, mean ± std) | Floor | Chosen hyperparameters |
|---|---|---|---|---|
| Gradient boosting | 0.727 | **0.727 ± 0.014** | 0.500 | `n_estimators=426`, `learning_rate=0.012`, `max_depth=3`, `min_samples_leaf=2`, `subsample=0.77` |
| HistGradientBoosting | 0.717 | 0.727 ± 0.016 | 0.500 | `max_iter=82`, `learning_rate=0.11`, `max_leaf_nodes=4`, `min_samples_leaf=130`, `l2_regularization=0.0007` |
| AdaBoost | 0.723 | 0.727 ± 0.015 | 0.500 | base `max_depth=4`, `n_estimators=55`, `learning_rate=0.17` |
| Oblique random forest | 0.685 | 0.727 ± 0.015 | 0.500 | `max_depth=12`, `min_samples_leaf=87`, `n_components=8` |
| Random forest | 0.682 | 0.725 ± 0.014 | 0.500 | `max_depth=23`, `min_samples_leaf=23`, `max_features=sqrt` |
| Logistic regression | 0.719 | 0.718 ± 0.014 | 0.500 | `C=0.0022` |
| Decision tree | 0.605 | 0.717 ± 0.017 | 0.500 | `max_depth=16`, `min_samples_leaf=88`, `criterion=gini` |
| Naive Bayes | 0.685 | 0.705 ± 0.012 | 0.500 | `var_smoothing=0.091` |
| Dummy (no model) | 0.500 | — | 0.500 | — |

Untuned params: `class_weight: balanced` for logistic regression, the tree, the random forest and
HistGradientBoosting; otherwise as in week 6, and `random_state=42` only for gradient boosting.

**Selection metric: ROC-AUC.** COMPAS is a risk score used to *rank* people, and ROC-AUC measures
exactly that -- the probability that a random reoffender gets a higher score than a random
non-reoffender -- without fixing a threshold, which is a separate decision (how many people to
flag) that I don't want the model choice to prejudge. It is also the steadier metric to compare
models on: it uses the whole ranking rather than which side of 0.5 each row falls on, so its
fold-to-fold differences are less noisy than accuracy's.

**What the board says.** With the corrected paired t-test on the same folds, the top four
(gradient boosting, HistGradientBoosting, AdaBoost, oblique forest) cannot be separated: gradient
boosting is ahead by 0.0004-0.001, winning 3 or 4 folds of 5, p > 0.7. The random forest is
behind by 0.002 in all 5 folds (p = 0.09, not significant). Logistic regression (-0.009,
p = 0.004), the decision tree (-0.011, p = 0.03) and Naive Bayes (-0.022, p = 0.004) are genuinely
worse. Tuning mattered most where the defaults overfit (the tree +0.11, both forests +0.04) and
almost not at all for gradient boosting, whose defaults are already shallow trees with a small
learning rate. I would report **gradient boosting, 0.727 ± 0.014 from nested CV**, knowing the
other three boosted/ensemble models are tied with it.

### Feature selection

Week 8 adds XGBoost and a feature-selection step. Both runs below are `python main.py` with
`model.type: xgboost`, tuning on (30 Optuna trials x 5 inner folds, nested over the same 5 outer
folds as every run on the scoreboard); the only difference is `feature_selection.enabled`. With
selection on, `select__min_votes` (1-3) is tuned together with XGBoost's nine hyperparameters, so
the nested score covers the choice of how much to select too.

| XGBoost, tuned | Untuned ROC-AUC (CV) | Tuned ROC-AUC (nested CV, mean ± std) | Fold scores | Features the final model uses |
|---|---|---|---|---|
| Without selection | 0.724 | **0.728 ± 0.015** | 0.733, 0.701, 0.733, 0.738, 0.734 | all 10 |
| With selection (`VoteSelector`) | 0.721 | 0.725 ± 0.013 | 0.724, 0.703, 0.732, 0.732, 0.734 | 2: `age`, `priors_count` |

Untuned = the `config.yaml` params (300 trees, learning rate 0.05, depth 4). **Corrected paired
t-test** (Nadeau & Bengio, same folds, `test_train_ratio = 1/4`): with minus without selection =
**-0.0024**, selection ahead in 2 of 5 folds, t = -0.82, **p = 0.46**. Not a significant
difference. Without selection, XGBoost ties last week's best, gradient boosting (+0.0002,
p = 0.93).

**The features kept.** The vote is unanimous in both directions: `age` and `priors_count` get all
three votes, and every other feature gets none -- the random forest and gradient boosting both
put their above-average importance on those two, and the L1 logistic regression (C = 0.01, a
strong penalty) shrinks everything else to zero. So on the full development set `min_votes` 1, 2
and 3 all keep the same two features, which is why tuning settles on 2 without it mattering. Inside
the outer folds it chose 2, 1, 2, 1 and 3: the choice moves around because the candidates are
nearly indistinguishable there too, not because one setting is better.

**Is the smaller model as good?** Yes, within the noise: two features out of ten keep the ROC-AUC
at 0.725 against 0.728, a gap of 0.002 that is a fifth of the fold-to-fold std and far from
significant (p = 0.46). Most of what these ten columns say about reoffending is in a person's age
and number of prior offences; the juvenile counts, sex, charge degree and the missingness flags add
at most a few thousandths. **Would I keep selection on?** Yes, and `config.yaml` leaves it on,
but for explainability rather than accuracy. It buys no score here -- there are no noise columns
for it to remove, unlike the notebook's case with 20 added random columns -- and it makes each run
slower, because the selector fits three models per fold. What it buys is a model that rests on two
inputs anyone can check, which on a dataset used to inform bail decisions is not a minor point,
with the same fairness picture (false-positive rate 0.30 for African-American and 0.16 for
Caucasian defendants, against 0.30 / 0.17 with all ten). If the goal were the last thousandth of
ROC-AUC, the honest answer is that the two runs are tied and either is defensible. I will re-check
it when engineered features (week 6 challenge 1) add columns that could be noise.

## Beyond COMPAS

Week 9, option A: the same pipeline on a different dataset, the Adult income data (1994 US Census,
UCI Adult, subsampled and extended for the course). The question is still yes/no: does this person
earn more than 50,000 USD a year (`income_over_50k`, about 24% yes)? Only the data changes, so
whatever breaks is a place where the pipeline quietly assumed COMPAS.

**How to run it.** `python main.py config_adult.yaml`. Everything specific to Adult lives in that
file: the data and target, the cleaning rules, the feature lists, its own `results/adult/` folder
(so its scoreboard never mixes with COMPAS's) and a `predict` section. `python main.py` still runs
COMPAS from `config.yaml`, with exactly the same numbers as before today (checked on the same folds
with tuning off: same fold scores, same reports).

**Choices in `config_adult.yaml`.**
- `sex` is the audited attribute. `race`, the other sensitive one, is left out of the features
  too, so the model uses neither protected characteristic directly.
- `fnlwgt` is dropped (a census sampling weight, not a property of the person) and `education` is
  a redundant column (one to one with `education_num`, which keeps the order).
- `"?"` is a placeholder token. `workclass` and `occupation` are missing for the same people, who
  earn more than 50K about half as often, so that gap is filled **and** flagged
  (`occupation_was_missing`, whose 381 rows include workclass's 379). `native_country` and
  `v01`-`v06` are missing with no pattern: imputed only.
- Adult's own validity rules: age 17-90 (COMPAS's minimum of 18 would erase 74 valid people),
  hours 1-99, capital gain and loss at least 0. The 99999 capital gains are the census's top code,
  so there is no maximum.
- All 25 undocumented `v` columns go in as features, and feature selection decides.
- Target encoding again: `native_country` has 41 values, most with fewer than 5 people.

**What broke, why, and how I fixed it.**

| What broke | Why | Fix |
|---|---|---|
| No way to run another dataset | `main.py` always loaded `config.yaml` | `main.py` takes the config file as an argument, `config.yaml` when none is given; the report labels name the config actually used |
| `"?"` survived as a category in `workclass`, `occupation` and `native_country` | `clean_dataset()` turned placeholders into missing values only inside `_canonicalize_categories()`, which only touches columns with a canonical map. On COMPAS those were all the text columns, so it never showed | Placeholders become missing values in every text column. Values are only replaced, never stripped or recased, so a clean column comes out unchanged. Missing counts after cleaning: 379, 381, 86, as expected |
| The fairness report crashed (no `score_text`) | `split_features_target()` and `fairness_report()` named COMPAS's own score in the code | The reference score is an optional `data.reference_score` in the config (`config.yaml` sets it to `score_text`; Adult has none). The report now speaks of class 0 and class 1, not of reoffending |
| No predictions for new people | the pipeline only ever evaluated | A `predict` section: the refit pipeline predicts `adult_new.csv` after the same cleaning, without `drop_duplicate_rows()`, using the training columns in the same order, and stops if rows out differ from rows in |

**Results.** XGBoost, tuned (30 Optuna trials x 5 inner folds, nested over 5 outer folds; the
nested scores are the honest ones). The only difference between the two runs is
`feature_selection.enabled`.

| XGBoost, tuned | ROC-AUC | Accuracy | F1 (class 1, 0.5) | Features the final model uses |
|---|---|---|---|---|
| Target | ≥ 0.90 | ≥ 0.855 | ≥ 0.66 | |
| No model (always "≤ 50K") | 0.50 | 0.76 | 0.00 | |
| Without selection | 0.906 ± 0.014 | 0.856 ± 0.011 | 0.654 ± 0.035 ✗ | all 36 |
| **With selection (`VoteSelector`)** | **0.908 ± 0.013** | **0.859 ± 0.009** | **0.672 ± 0.034** | 8 (below) |

With selection, all three targets are met in the same run. Without it, F1 misses (0.654).
**Corrected paired t-test** on ROC-AUC (same folds, `test_train_ratio = 1/4`): with minus without
selection = +0.0023, selection ahead in 4 of 5 folds, t = 1.32, p = 0.26. So on ROC-AUC the two are
tied, and the difference that matters is F1 (0.672 against 0.654), where selection keeps the
recall up (0.61 against 0.57). Tuning chose `min_votes` 1 on the full development set and 1, 2, 2,
1, 1 in the outer folds, which keeps the same features, because every feature has either 0 or at
least 2 votes. Untuned, with selection, the run already reaches 0.913 / 0.864 / 0.686 (plain CV,
not nested).

**Features kept (8 of 36):** `age`, `education_num`, `capital_gain` and `relationship` with all
three votes; `capital_loss`, `hours_per_week`, `marital_status` and `occupation` with two. **None
of the 25 `v` columns gets a single vote**, and neither do `workclass`, `native_country` or the
missingness flag (what it says is already in `occupation`).

**Fairness, by sex** (out-of-fold, share of people earning at most 50K who were predicted to earn
more): 0.02 for women and 0.09 for men with selection, 0.01 / 0.08 without. Here a false positive
means overestimating someone's income, and it happens about four times as often to men.

**Predictions.** `predictions/adult_new_predictions.csv`: `id`, probability and 0/1 prediction
(threshold 0.5) for the 2,000 people in `adult_new.csv`, one row each; 19.3% are predicted to earn
more than 50K.

**What I learned.** The hardest assumption to find was the placeholder one: nothing crashed. The
`"?"` simply became a category of its own, and the pipeline ran happily with it. It was invisible on
COMPAS because the columns with a canonical map happened to be all of its text columns; only
checking `value_counts(dropna=False)` after cleaning showed it. The other two failures announced
themselves with an error. What got the model to the targets was not tuning (it adds nothing over
the defaults here) but the cleaning and above all feature selection: the 25 undocumented columns
are noise, and removing them is what took F1 over 0.66 in these runs. The gain is modest (0.018,
about half the fold-to-fold spread), but it is the first time selection has had something to
remove: COMPAS in week 8 had no noise columns.

## Missing values

Week 10: which imputer fills the numeric gaps is now a setting,
`preprocessing.imputation.numeric_strategy`. Four runs of `python main.py` on COMPAS, identical
except for the imputer: XGBoost with the `config.yaml` params, feature selection on, tuning off,
the same 5 folds as every run on the scoreboard.

| Imputer | ROC-AUC (CV, mean ± std) | Fold scores | vs median: difference, folds ahead, p | Run time |
|---|---|---|---|---|
| `median` | 0.7207 ± 0.013 | 0.718, 0.700, 0.727, 0.725, 0.733 | | 11 s |
| `knn` (k = 5) | 0.7200 ± 0.013 | 0.718, 0.699, 0.725, 0.726, 0.732 | -0.0007, 2/5, p = 0.28 | 12 s |
| `iterative` (MICE) | 0.7208 ± 0.014 | 0.721, 0.698, 0.729, 0.723, 0.733 | +0.0001, 2/5, p = 0.95 | 12 s |
| `random_forest` (missForest) | 0.7209 ± 0.013 | 0.721, 0.699, 0.728, 0.724, 0.733 | +0.0002, 2/5, p = 0.88 | 135 s |

The comparison is the **corrected paired t-test** (Nadeau & Bengio, same folds,
`test_train_ratio = 1/4`).

**Which imputer I keep: `median`.** The four are tied: the largest difference is seven ten-thousandths, about a twentieth of the fold-to-fold
std, and no p-value is anywhere near significance. The model-based imputers have little to work
with here. After cleaning, the numeric gaps are in `priors_count` (507), `juv_fel_count` (222)
and `age` (155), and each one can only be guessed from the other four numeric columns, which are
weakly related to it (age says little about a person's number of priors). The biggest gap
also matters least for the guess itself: `priors_count_was_missing` already flags it, so the model
sees that the value was missing whatever it was filled with. So the median stays:
it is the simplest, the fastest (the forest imputer is twelve times slower for nothing), and the
easiest to explain for a model that ends up using two features.

## Predicting new data

`python main.py` now saves the final pipeline (refit on all development rows, tuned if tuning is
on) to `models/model.joblib`. Then, with no training:

```bash
python predict.py data/compas_new_people.csv predictions.csv
python predict.py data/adult_new.csv adult_predictions.csv --config config_adult.yaml   # the Adult model
```

The input needs the training data's columns but no target. It goes through the same fit-free
cleaning as the training file (placeholders, impossible values, category spellings), with no
duplicate dropping. Everything learned (the medians, the encodings, the scale, the features kept,
the model) comes from the saved file, so it is exactly what `main.py` fitted.

The output, `predictions.csv`, has one row per input row (20 for `compas_new_people.csv`) and three
columns: `id`, `probability` (the predicted probability of reoffending within two years) and
`prediction` (1 when that probability is at least 0.5). With the model `python main.py` saves from
`config.yaml` (tuned XGBoost on `age` and `priors_count`, nested ROC-AUC 0.725 ± 0.013, the same
as week 8), 6 of the 20 new people are predicted to reoffend.

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
python main.py                      # COMPAS, config.yaml
python main.py config_adult.yaml    # week 9: Adult income (see "Beyond COMPAS")
python predict.py data/compas_new_people.csv predictions.csv   # week 10: predict with the saved model (see "Predicting new data")
```

This loads `config.yaml`, cleans the data, sets the locked test set aside, cross-validates the
pipeline on the development set, and prints:
- the **selection metric** (`cv.selection_metric`, ROC-AUC) of the default model -- mean ± std over
  the 5 folds, the train-validation gap (a large, consistent gap is overfitting) and the
  **no-model floor** (a dummy on the same folds)
- **every metric** in `cv.metrics`, mean ± std
- the **scoreboard**: every run so far on the same data and metric, best first
- and saves to the results file the per-fold tables, a **confusion matrix and classification
  report on the out-of-fold predictions** (threshold 0.5), so every development row is scored by a
  model that never trained on it
- a **false-positive-rate-by-race comparison** between our model and COMPAS's own score, on those
  same out-of-fold predictions

With `tuning.enabled: true` (the default since week 5) it also prints the nested CV score of the
tuned model and the chosen hyperparameters; the results file gets the **nested CV table** (one row
per outer fold, every metric, the hyperparameters chosen in it) and the **tuning report** (top 5
trials and the optimism check). The metrics, classification and fairness reports then use the
nested out-of-fold predictions, the scoreboard row the nested fold scores, and the final refit the
tuned hyperparameters. With `enabled: false` there is no tuning. Logistic regression, Naive Bayes
and the decision tree take a minute or two; the ensembles and boosters 10-20 minutes each (900
fits), so while testing set `tuning.n_trials: 5` or `tuning.enabled: false`.

It then refits the pipeline on the whole development set -- cross-validation estimates how good
the *recipe* is and throws its five models away; this last fit is the model you would actually
use. The locked test set is never scored.

All of this is also saved to a timestamped file in `results/` (e.g. `results/run_20261003_191844.txt`),
so it doesn't just scroll past in your terminal, and every run adds one row to
`results/runs.csv` (the scoreboard: model, params, tuned params, mean, std, untuned score, floor and
the score on each fold). Only runs on the same data, cleaning and folds are comparable, so delete
`results/runs.csv` when any of those change. `results/` is created automatically the first time you
run the pipeline, and isn't tracked in git (see `.gitignore`) since it's generated output, not source.

To reproduce the scoreboard above, switch `model.type` in `config.yaml` between `dummy`,
`logistic_regression`, `naive_bayes`, `decision_tree`, `random_forest`, `oblique_random_forest`,
`adaboost`, `gradient_boosting`, `hist_gradient_boosting` and `xgboost` (the matching `params` are
commented there) and run `python main.py` for each (`tuning.enabled: false` for the dummy). The
week 7 rows were run before feature selection existed: set `feature_selection.enabled: false` to
reproduce them. With it on, the output also ends with **Features kept**: the features the final
model uses and each one's votes.

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
