# DC Property ML

**A clean, modular R machine learning project for DC residential property price prediction.**

> Refactored from *Stat 627 Final Project Report* (Aaron Niecestro, December 2019).  
> Original dataset: [Kaggle — DC Residential Properties](https://www.kaggle.com/)

---

## Project Overview

This project builds and evaluates **12 machine learning models** on ~57,610 DC residential
property records to answer three research questions:

1. What are the best variables to explain property pricing in DC?
2. Do classification models better explain the data than linear/Bayesian baselines?
3. How trustworthy are the classification models?

**Response variable:** `Price_10K` — sale price in units of $10,000 (mean = $577,250)

---

## Project Structure

```
dc-property-ml/
├── config/
│   └── config.yaml              # Master config: paths, hyperparameters, thresholds
├── data/
│   ├── raw/                     # Place DC_Properties.csv here
│   ├── processed/               # dc_clean.rds (generated)
│   └── splits/                  # train.rds / test.rds (generated)
├── src/
│   ├── 00_install_packages.R    # One-time package installer
│   ├── 01_load_config.R         # YAML config loader
│   ├── 02_load_data.R           # Raw CSV → cleaned tibble
│   ├── 03_split_data.R          # Unified 50/50 train/test split + class labels
│   ├── 04_feature_engineering.R # Polynomial terms + interaction features
│   ├── 05_models_linear.R       # Linear regression + variable selection
│   ├── 06_models_polynomial.R   # Polynomial regression + outlier detection
│   ├── 07_models_knn.R          # KNN (k-loop 1–100 + best-k evaluation)
│   ├── 08_models_lda_qda.R      # LDA & QDA with/without equal priors
│   ├── 09_models_ridge_lasso.R  # Ridge & Lasso via glmnet
│   ├── 10_models_pcr_pls.R      # PCR & PLS via pls package
│   ├── 11_models_trees.R        # Regression tree + cost-complexity pruning
│   ├── 12_models_random_forest.R# Random Forest, optimal RF, Bagging
│   ├── 13_models_svm.R          # SVM (4 kernels, median-split classification)
│   └── 14_evaluate.R            # Unified metrics, comparison tables, plots
├── utils/
│   ├── utils_io.R               # Model save/load, run log, directory setup
│   ├── utils_plots.R            # EDA + diagnostic plot functions
│   └── utils_preprocessing.R   # Data quality report, numeric summary, scaling
├── models/                      # Serialised .rds model files (generated)
├── notebooks/
│   ├── 01_eda.Rmd               # Exploratory Data Analysis notebook
│   └── 02_modeling.Rmd          # Full modelling notebook (all 12 models)
├── outputs/
│   ├── plots/                   # All generated PNG plots
│   ├── tables/                  # CSV comparison tables + run_log.csv
│   └── reports/                 # Rendered HTML reports
├── run_all.R                    # ★ End-to-end pipeline runner
├── dc-property-ml.Rproj         # RStudio project file
└── .Rprofile                    # Auto-loads packages on project open
```

---

## Quick Start

### 1. Install packages (once)
```r
source("src/00_install_packages.R")
```

### 2. Add the raw data
```
data/raw/DC_Properties.csv
```

### 3. Run the full pipeline
```r
source("run_all.R")
```

Or skip data processing if splits already exist:
```bash
Rscript run_all.R --skip-data
```

### 4. Knit the notebooks interactively
Open in RStudio and knit:
- `notebooks/01_eda.Rmd`
- `notebooks/02_modeling.Rmd`

---

## Models Implemented

### Regression (response: `Price_10K`)

| Model | Key Setting | Paper Adj. R² / RSE |
|---|---|---|
| Multiple Linear Regression | AIC/BIC stepwise selection | 64.99% / 33.49 |
| Polynomial Regression | Degree 8 (Bathrooms, Bedrooms); Degree 5 (AYB_age) | 69.04% / 31.12 |
| Ridge Regression | α=0; 10-fold CV lambda | — |
| Lasso Regression | α=1; 10-fold CV lambda | — |
| PCR | 29 components (min CV RMSEP) | — |
| PLS | ~3 components (RMSEP stabilises) | — |
| Regression Tree (Pruned) | CP-optimised; 11 terminal nodes | — |
| Random Forest (Full) | 500 trees; Longitude most important | — |
| Random Forest (Optimal) | 143 trees; mtry=1; 54.81% var explained | — |
| Bagging | 500 trees; mtry=p | — |

### Classification

| Model | Target | Threshold | Paper Correct Rate |
|---|---|---|---|
| KNN | Reasonable / Overpriced | Mean = 57.725 | 0.5277 (k=1) |
| LDA (equal priors) | Reasonable / Overpriced | Mean = 57.725 | 0.8093 |
| QDA (equal priors) | Reasonable / Overpriced | Mean = 57.725 | 0.8061 |
| SVM (radial) | Under / Over | Median = 44.365 | 0.6385 (misclass) |

---

## Predictors

| Variable | Type | Description |
|---|---|---|
| `Price_10K` | Numeric (response) | Sale price in $10,000 units |
| `Bathrooms` | Numeric | Full bathrooms (0–11; mean 2.52) |
| `Rooms` | Numeric | Total rooms (0–30; mean 7.44) |
| `Bedrooms` | Numeric | Bedrooms (0–15; mean 3.42) |
| `Kitchens` | Numeric | Kitchens (0–6; mean 1.25) |
| `Fireplaces` | Numeric | Fireplaces (0–9; mean 0.64) |
| `Stories` | Numeric | Number of stories |
| `AYB_age` | Numeric | Years since actual year built (mean 84.6) |
| `EYB_age` | Numeric | Years since effective year built (mean 49.2) |
| `Remodel_age` | Numeric | Years since last remodel (mean 8.4) |
| `Latitude` | Numeric | Property latitude |
| `Longitude` | Numeric | Property longitude |
| `Condition` | Factor | Property condition rating |
| `Grade` | Factor | Construction quality grade |
| `Ward` | Factor | DC ward (1–8) |
| `AC` | Factor | Air conditioning (Yes/No) |
| `Qualified` | Factor | DC housing qualification status |

---

## Data Processing Pipeline

```
DC_Properties.csv (158,957 rows × 49 cols)
        ↓
Filter: Price > $10K, Rooms ≤ 40
Fill NAs → 0
Derive AYB_age, EYB_age, Remodel_age
Select 21 analysis columns
        ↓
dc_clean.rds (57,610 rows × 21 cols)
        ↓
50/50 random split (seed = 627)
        ↓
train.rds (28,805 rows)    test.rds (28,805 rows)
        ↓
add_polynomial_features()   ← Bathrooms^8, AYB_age^5, etc.
build_Xy()                  ← numeric model.matrix for glmnet
add_classification_labels() ← mean/median threshold splits
```

---

## Configuration

All hyperparameters, thresholds, and paths live in `config/config.yaml`.
No hardcoded values appear in any `.R` source file.

Key settings:
```yaml
project:
  seed: 627                    # reproducibility
split:
  ratio: 0.50                  # 50/50 train/test
classification:
  mean_threshold:   57.725     # Reasonable vs Overpriced
  median_threshold: 44.365     # Under vs Over
polynomial:
  Bathrooms_deg: 8
  AYB_age_deg:   5
models:
  knn:
    k_loop_max: 100
  random_forest:
    ntree_values: [143, 400, 500]
```

---

## Key Findings (from paper)

1. **Best predictors**: Longitude, Bathrooms, Fireplaces, Bedrooms, AYB_age, Grade
2. **Polynomial model** (Adj. R² = 69.04%) outperforms linear (64.99%)
3. **LDA** achieves the highest correct classification rate: **80.93%**
4. **Classification models** outperform the earlier linear/logistic baselines
5. **Random Forest (optimal)**: Bathrooms is the most important predictor; Kitchens least
6. All VIFs < 5 — no multicollinearity concern

---

## Dependencies

| Package | Purpose |
|---|---|
| `dplyr`, `tidyr`, `readr`, `purrr` | Data wrangling |
| `yaml`, `here` | Config & project paths |
| `MASS` | LDA, QDA, stepAIC |
| `glmnet` | Ridge & Lasso |
| `pls` | PCR & PLS |
| `class` | KNN |
| `rpart`, `rpart.plot` | Regression trees |
| `randomForest` | Random Forest & Bagging |
| `e1071` | SVM |
| `car` | VIF, outlier test |
| `ggplot2`, `GGally` | Visualisation |
| `rmarkdown`, `knitr` | Notebooks |

---

*DC Property ML — Refactored from Stat 627 Final Project | Aaron Niecestro*
