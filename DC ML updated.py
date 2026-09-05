> “This is a conversion of previous R code in to python working code.  
> This file is for my machine learning work of DC Property Data from Kaggle to practice converting work from R to python.”

```python
# -*- coding: utf-8 -*-
"""
Created on February 13, 2026
Edited on February 13, 2026

Modernized version of the original DC Property Machine Learning script.
R → Python conversion, now using:
- pathlib for paths
- consistent type hints
- sklearn Pipelines for modeling
- cleaner modular structure
"""

from __future__ import annotations

# ============================================================
# MODULE 0 — IMPORTS & GLOBAL SETUP
# ============================================================

from pathlib import Path
import warnings
import itertools
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import anova_lm
import statsmodels.genmod.generalized_linear_model as glm

from sklearn.metrics import (
    roc_curve,
    auc,
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    BaggingRegressor,
    BaggingClassifier,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
glm.SET_USE_BIC_LLF(True)


# ============================================================
# MODULE 1 — DATA LOADING & CLEANING
# ============================================================

def load_dc_data(path: Path) -> pd.DataFrame:
    """
    Load DC property data from Excel and perform initial cleaning.
    """
    dc = pd.read_excel(path)

    # Replace NA with 0 (as in original R code)
    dc = dc.fillna(0)

    # Select relevant columns
    cols = [
        "PRICE", "BATHRM", "HF_BATHRM", "HEAT", "AC", "ROOMS", "BEDRM",
        "AYB", "YR_RMDL", "EYB", "STORIES", "QUALIFIED", "GRADE", "CNDTN",
        "KITCHENS", "FIREPLACES", "WARD", "QUADRANT", "LATITUDE", "LONGITUDE",
    ]
    dc = dc[cols].copy()

    # Numeric columns
    num_cols = [
        "PRICE", "BATHRM", "HF_BATHRM", "ROOMS", "BEDRM", "AYB", "YR_RMDL",
        "EYB", "STORIES", "KITCHENS", "FIREPLACES", "LATITUDE", "LONGITUDE",
    ]
    dc[num_cols] = dc[num_cols].apply(pd.to_numeric, errors="coerce")

    # Categorical columns
    cat_cols = ["HEAT", "AC", "QUALIFIED", "GRADE", "CNDTN", "WARD", "QUADRANT"]
    dc[cat_cols] = dc[cat_cols].astype("string")

    # Filter rows (Python equivalent of R filters)
    dc = dc[
        (~dc["CNDTN"].isin(["", "Default", "Poor"])) &
        (dc["GRADE"] != " No Data") &
        (dc["GRADE"] != "") &
        (dc["HEAT"] != "No Data") &
        (dc["PRICE"].between(10000, 10000000)) &
        (dc["FIREPLACES"] < 10) &
        (dc["KITCHENS"] <= 10) &
        (dc["ROOMS"] <= 40) &
        (dc["BEDRM"] <= 20) &
        (dc["STORIES"] <= 10) &
        (dc["LATITUDE"] != 0) &
        (dc["LONGITUDE"] != 0)
    ].copy()

    # Age variables
    dc["AYB_age"] = np.where(dc["AYB"] == 2019, 0, 2019 - dc["AYB"])
    dc["EYB_age"] = np.where(dc["EYB"] == 2019, 0, 2019 - dc["EYB"])

    # Remodel age
    dc["REMODEL_age"] = dc["YR_RMDL"].replace({"0": 0, "20": 0}).astype(float)
    dc["REMODEL_age"] = np.where(dc["REMODEL_age"] == 0, 0, 2019 - dc["REMODEL_age"])

    # Collapse Exceptional grades
    dc["GRADE"] = dc["GRADE"].replace({
        "Exceptional-A": "Exceptional",
        "Exceptional-B": "Exceptional",
        "Exceptional-C": "Exceptional",
        "Exceptional-D": "Exceptional",
    })

    # Binary qualification
    dc["QUALIFIED_2"] = (dc["QUALIFIED"] == "Q").astype(int)

    # Fix AC
    dc["AC"] = dc["AC"].replace({"0": "N"})

    # Drop unused columns
    dc = dc.drop(columns=["CNDTN", "AYB", "YR_RMDL", "EYB"])

    # Final cleaning
    dc = dc.dropna()
    dc = dc[dc["AYB_age"] < 2000]

    # Create PRICE_10K
    dc["PRICE_10K"] = dc["PRICE"] / 10000

    # Combine bathrooms
    dc["BATHRM"] = dc["BATHRM"] + 0.5 * dc["HF_BATHRM"]

    # Final column order
    dc = dc[
        [
            "PRICE", "PRICE_10K", "BATHRM", "ROOMS", "BEDRM", "STORIES",
            "KITCHENS", "FIREPLACES", "LATITUDE", "LONGITUDE",
            "AYB_age", "EYB_age", "REMODEL_age",
            "HEAT", "AC", "QUALIFIED", "QUALIFIED_2", "GRADE",
            "WARD", "QUADRANT",
        ]
    ].copy()

    return dc


def train_test_split_dc(dc: pd.DataFrame, n_sample: int = 57610, seed: int = 10_000_000
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample a fixed-size dataset and split into train/test.
    """
    np.random.seed(seed)
    dc_sampled = dc.sample(n_sample, replace=True, random_state=seed)

    train, test = train_test_split(dc_sampled, test_size=0.5, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ============================================================
# MODULE 2 — DIAGNOSTICS (QQ PLOTS, HISTOGRAMS)
# ============================================================

def plot_diagnostics(dc: pd.DataFrame, vars_: List[str]) -> None:
    """
    Plot QQ plots and histograms for selected variables.
    """
    import scipy.stats as stats

    # QQ plots
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for ax, var in zip(axes, vars_):
        stats.probplot(dc[var], dist="norm", plot=ax)
        ax.set_title(f"QQ Plot — {var}")
    plt.tight_layout()
    plt.show()

    # Histograms
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for ax, var in zip(axes, vars_):
        sns.histplot(dc[var], kde=True, ax=ax)
        ax.set_title(f"Histogram — {var}")
    plt.tight_layout()
    plt.show()


# ============================================================
# MODULE 3 — STEPWISE MODEL SELECTION (AIC / BIC)
# ============================================================

def _get_ic(model, criterion: str = "aic") -> float:
    """
    Return chosen information criterion from a fitted statsmodels model.
    """
    criterion = criterion.lower()
    if criterion == "aic":
        return model.aic
    elif criterion == "bic":
        return getattr(model, "bic_llf", model.bic)
    else:
        raise ValueError("criterion must be 'aic' or 'bic'")


def forward_stepwise(
    data: pd.DataFrame,
    response: str,
    candidates: List[str],
    family,
    criterion: str = "aic",
):
    """
    Forward stepwise selection based on AIC or BIC.
    """
    selected: List[str] = []
    best_ic = np.inf
    best_model = None
    best_formula: Optional[str] = None
    remaining = candidates.copy()

    improved = True
    while improved and remaining:
        improved = False
        ic_candidates = []

        for term in remaining:
            formula_terms = selected + [term]
            formula = response + " ~ " + " + ".join(formula_terms)
            try:
                model = smf.glm(formula=formula, data=data, family=family).fit()
                ic = _get_ic(model, criterion)
                ic_candidates.append((ic, term, model, formula))
            except Exception:
                continue

        if not ic_candidates:
            break

        ic_candidates.sort(key=lambda x: x[0])
        best_candidate_ic, best_term, candidate_model, candidate_formula = ic_candidates[0]

        if best_candidate_ic + 1e-8 < best_ic:
            best_ic = best_candidate_ic
            best_model = candidate_model
            best_formula = candidate_formula
            selected.append(best_term)
            remaining.remove(best_term)
            improved = True

    return best_model, best_formula, selected, best_ic


def backward_stepwise(
    data: pd.DataFrame,
    response: str,
    initial_terms: List[str],
    family,
    criterion: str = "aic",
):
    """
    Backward stepwise selection based on AIC or BIC.
    """
    selected = initial_terms.copy()
    formula = response + " ~ " + " + ".join(selected)

    model = smf.glm(formula=formula, data=data, family=family).fit()
    best_ic = _get_ic(model, criterion)
    best_model = model
    best_formula = formula

    improved = True
    while improved and len(selected) > 1:
        improved = False
        ic_candidates = []

        for term in selected:
            trial_terms = [t for t in selected if t != term]
            formula = response + " ~ " + " + ".join(trial_terms)
            try:
                model = smf.glm(formula=formula, data=data, family=family).fit()
                ic = _get_ic(model, criterion)
                ic_candidates.append((ic, term, model, formula))
            except Exception:
                continue

        if not ic_candidates:
            break

        ic_candidates.sort(key=lambda x: x[0])
        best_candidate_ic, removed_term, candidate_model, candidate_formula = ic_candidates[0]

        if best_candidate_ic + 1e-8 < best_ic:
            best_ic = best_candidate_ic
            best_model = candidate_model
            best_formula = candidate_formula
            selected.remove(removed_term)
            improved = True

    return best_model, best_formula, selected, best_ic


def all_subsets_ic(
    data: pd.DataFrame,
    response: str,
    candidates: List[str],
    family,
    criterion: str = "aic",
):
    """
    All-subsets model selection based on AIC or BIC.
    """
    best_ic = np.inf
    best_model = None
    best_formula = None
    best_subset = None

    for k in range(1, len(candidates) + 1):
        for subset in itertools.combinations(candidates, k):
            formula = response + " ~ " + " + ".join(subset)
            try:
                model = smf.glm(formula=formula, data=data, family=family).fit()
                ic = _get_ic(model, criterion)
                if ic < best_ic:
                    best_ic = ic
                    best_model = model
                    best_formula = formula
                    best_subset = list(subset)
            except Exception:
                continue

    return best_model, best_formula, best_subset, best_ic


# ============================================================
# MODULE 4 — BEST SUBSET SELECTION (regsubsets-style)
# ============================================================

def compute_metrics(model, n: int) -> dict:
    """
    Compute AIC, BIC, Adjusted R², Cp for a fitted statsmodels model.
    """
    rss = np.sum(model.resid ** 2)
    k = model.df_model + 1  # includes intercept
    sigma2 = rss / (n - k)

    return {
        "AIC": model.aic,
        "BIC": getattr(model, "bic_llf", model.bic),
        "Adj_R2": getattr(model, "rsquared_adj", np.nan),
        "Cp": rss / sigma2 - (n - 2 * k),
    }


def best_subset_selection(
    data: pd.DataFrame,
    response: str,
    predictors: List[str],
    max_features: Optional[int] = None,
) -> pd.DataFrame:
    """
    Perform best subset selection (like regsubsets in R).
    """
    if max_features is None:
        max_features = len(predictors)

    results = []
    n = data.shape[0]

    for k in range(1, max_features + 1):
        for subset in itertools.combinations(predictors, k):
            formula = response + " ~ " + " + ".join(subset)
            try:
                model = smf.ols(formula=formula, data=data).fit()
                metrics = compute_metrics(model, n)
                results.append(
                    {
                        "subset": subset,
                        "k": k,
                        "AIC": metrics["AIC"],
                        "BIC": metrics["BIC"],
                        "Adj_R2": metrics["Adj_R2"],
                        "Cp": metrics["Cp"],
                        "model": model,
                        "formula": formula,
                    }
                )
            except Exception:
                continue

    return pd.DataFrame(results)


def plot_subset_metric(results_df: pd.DataFrame, metric: str = "BIC") -> None:
    """
    Plot model size vs. metric (AIC, BIC, Adj_R2, Cp).
    """
    plt.figure(figsize=(8, 5))
    grouped = results_df.groupby("k")[metric].min()
    plt.plot(grouped.index, grouped.values, marker="o")
    plt.xlabel("Number of Predictors")
    plt.ylabel(metric)
    plt.title(f"Best Subset Selection — {metric}")
    plt.grid(True)
    plt.show()


# ============================================================
# MODULE 5 — PARTIAL F-TESTS (ANOVA) FOR NESTED MODELS
# ============================================================

def partial_f_test(
    data: pd.DataFrame,
    full_formula: str,
    reduced_formula: str,
):
    """
    Perform a partial F-test comparing a full model vs. a reduced model.
    """
    full_model = smf.ols(full_formula, data=data).fit()
    reduced_model = smf.ols(reduced_formula, data=data).fit()
    anova_results = anova_lm(reduced_model, full_model)
    return anova_results, full_model, reduced_model


# ============================================================
# MODULE 6 — POLYNOMIAL REGRESSION
# ============================================================

def poly_regression_statsmodels(
    data: pd.DataFrame,
    response: str,
    predictor: str,
    degree: int,
):
    """
    Fit polynomial regression using statsmodels formula interface.
    """
    terms = " + ".join([f"np.power({predictor}, {d})" for d in range(1, degree + 1)])
    formula = f"{response} ~ {terms}"
    model = smf.ols(formula=formula, data=data).fit()
    return model, formula


def poly_regression_sklearn(
    X: pd.DataFrame,
    y: np.ndarray,
    degree: int,
) -> Pipeline:
    """
    Fit polynomial regression using sklearn Pipeline.
    """
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )
    model.fit(X, y)
    return model


# ============================================================
# MODULE 7 — RIDGE & LASSO REGRESSION
# ============================================================

def build_xy(
    data: pd.DataFrame,
    response: str,
    predictors: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build X and y matrices for sklearn models.
    """
    X = data[predictors].copy()
    y = data[response].values
    return X, y


def ridge_regression_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    alphas: Optional[np.ndarray] = None,
) -> Pipeline:
    """
    Fit Ridge regression with cross-validation.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 200)

    ridge_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas, store_cv_values=True)),
        ]
    )
    ridge_model.fit(X_train, y_train)
    return ridge_model


def lasso_regression_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    alphas: Optional[np.ndarray] = None,
) -> Pipeline:
    """
    Fit LASSO regression with cross-validation.
    """
    if alphas is None:
        alphas = np.logspace(-3, 1, 200)

    lasso_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lasso", LassoCV(alphas=alphas, cv=10, max_iter=5000)),
        ]
    )
    lasso_model.fit(X_train, y_train)
    return lasso_model


# ============================================================
# MODULE 8 — PCR & PLS
# ============================================================

def pcr_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
) -> Tuple[Pipeline, float]:
    """
    Principal Components Regression (PCR).
    """
    pcr_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("linreg", LinearRegression()),
        ]
    )
    pcr_model.fit(X_train, y_train)
    preds = pcr_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return pcr_model, mse


def scree_plot(X: pd.DataFrame) -> None:
    """
    Scree plot for PCA components.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA().fit(X_scaled)

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Scree Plot — PCA")
    plt.grid(True)
    plt.show()


def pls_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
) -> Tuple[PLSRegression, float]:
    """
    Partial Least Squares Regression.
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    preds = pls.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return pls, mse


# ============================================================
# MODULE 9 — KNN CLASSIFICATION
# ============================================================

def build_price_classes(
    data: pd.DataFrame,
    response: str = "PRICE_10K",
) -> np.ndarray:
    """
    Create price categories: Reasonable vs Expensive (median split).
    """
    median_price = data[response].median()
    labels = np.where(data[response] < median_price, "Reasonable", "Expensive")
    return labels


def knn_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit KNN classifier with scaling.
    """
    knn_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    knn_model.fit(X_train, y_train)
    preds = knn_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return preds, acc, cm


# ============================================================
# MAIN EXECUTION (EXAMPLES)
# ============================================================

if __name__ == "__main__":
    # Path to data (update as needed)
    data_path = Path(
        r"C:\Users\aniec\Mirror\Programming Projects\2019.01_2024.05 DC Property Composite Analysis\2019.05.20 R - DC Property using GLM\Data\DC_Properties.xlsx"
    )

    # Load and split data
    DC = load_dc_data(data_path)
    DC_train, DC_test = train_test_split_dc(DC)

    # Diagnostics
    diag_vars = [
        "BATHRM", "ROOMS", "BEDRM", "STORIES",
        "KITCHENS", "FIREPLACES",
        "AYB_age", "EYB_age", "REMODEL_age",
    ]
    plot_diagnostics(DC_train, diag_vars)

    # Partial F-test example
    full_formula = (
        "PRICE_10K ~ BATHRM + ROOMS + BEDRM + STORIES + QUALIFIED + GRADE + "
        "KITCHENS + FIREPLACES + WARD + LATITUDE + LONGITUDE + "
        "AYB_age + EYB_age + REMODEL_age + CONDITION"
    )
    reduced_formula = (
        "PRICE_10K ~ BATHRM + ROOMS + BEDRM + STORIES + QUALIFIED + GRADE + "
        "KITCHENS + FIREPLACES + WARD + LATITUDE + LONGITUDE + "
        "AYB_age + EYB_age + REMODEL_age"
    )

    # CONDITION may not exist in this cleaned version; guard accordingly
    if "CONDITION" in DC_train.columns:
        anova_table, full_model, reduced_model = partial_f_test(
            data=DC_train,
            full_formula=full_formula,
            reduced_formula=reduced_formula,
        )
        print("\n===== PARTIAL F-TEST =====")
        print(anova_table)

    # Polynomial regression examples
    model_bathrm, formula_bathrm = poly_regression_statsmodels(
        data=DC_train,
        response="PRICE_10K",
        predictor="BATHRM",
        degree=8,
    )
    print("\n===== Polynomial Regression — BATHRM (deg 8) =====")
    print(model_bathrm.summary())

    model_bedrm, formula_bedrm = poly_regression_statsmodels(
        data=DC_train,
        response="PRICE_10K",
        predictor="BEDRM",
        degree=8,
    )
    print("\n===== Polynomial Regression — BEDRM (deg 8) =====")
    print(model_bedrm.summary())

    X_ayb = DC_train[["AYB_age"]]
    y_ayb = DC_train["PRICE_10K"].values
    poly_model_ayb = poly_regression_sklearn(X_ayb, y_ayb, degree=4)
    print("\n===== Polynomial Regression (sklearn) — AYB_age (deg 4) =====")
    print("Coefficients:", poly_model_ayb.named_steps["linreg"].coef_)
    print("Intercept:", poly_model_ayb.named_steps["linreg"].intercept_)

    # Ridge & LASSO
    predictors_reg = [
        "BATHRM", "ROOMS", "BEDRM", "STORIES", "KITCHENS",
        "FIREPLACES", "LATITUDE", "LONGITUDE",
        "AYB_age", "EYB_age", "REMODEL_age",
    ]
    response_reg = "PRICE_10K"

    X_train_reg, y_train_reg = build_xy(DC_train, response_reg, predictors_reg)
    X_test_reg, y_test_reg = build_xy(DC_test, response_reg, predictors_reg)

    ridge_model = ridge_regression_cv(X_train_reg, y_train_reg)
    ridge_alpha = ridge_model.named_steps["ridge"].alpha_
    ridge_coef = ridge_model.named_steps["ridge"].coef_
    ridge_pred = ridge_model.predict(X_test_reg)
    ridge_mse = mean_squared_error(y_test_reg, ridge_pred)

    print("\n===== RIDGE REGRESSION =====")
    print("Optimal alpha:", ridge_alpha)
    print("Test MSE:", ridge_mse)

    lasso_model = lasso_regression_cv(X_train_reg, y_train_reg)
    lasso_alpha = lasso_model.named_steps["lasso"].alpha_
    lasso_coef = lasso_model.named_steps["lasso"].coef_
    lasso_pred = lasso_model.predict(X_test_reg)
    lasso_mse = mean_squared_error(y_test_reg, lasso_pred)

    print("\n===== LASSO REGRESSION =====")
    print("Optimal alpha:", lasso_alpha)
    print("Test MSE:", lasso_mse)

    # PCR & PLS
    X_train_pcr = DC_train[predictors_reg].values
    y_train_pcr = DC_train[response_reg].values
    X_test_pcr = DC_test[predictors_reg].values
    y_test_pcr = DC_test[response_reg].values

    pcr_model, pcr_mse = pcr_regression(
        X_train_pcr, y_train_pcr, X_test_pcr, y_test_pcr, n_components=5
    )
    print("\n===== PCR RESULTS =====")
    print("Test MSE:", pcr_mse)
    print("Explained variance (first 5 PCs):")
    print(pcr_model.named_steps["pca"].explained_variance_ratio_)

    print("\n===== Scree Plot (PCA) =====")
    scree_plot(DC_train[predictors_reg])

    pls_model, pls_mse = pls_regression(
        X_train_pcr, y_train_pcr, X_test_pcr, y_test_pcr, n_components=5
    )
    print("\n===== PLS RESULTS =====")
    print("Test MSE:", pls_mse)
    print("PLS Coefficients:")
    print(pls_model.coef_)

    # KNN classification
    X_train_knn = DC_train[predictors_reg].values
    X_test_knn = DC_test[predictors_reg].values
    y_train_knn = build_price_classes(DC_train)
    y_test_knn = build_price_classes(DC_test)

    preds_5, acc_5, cm_5 = knn_classify(X_train_knn, y_train_knn, X_test_knn, y_test_knn, k=5)
    print("\n===== KNN (k = 5) =====")
    print("Accuracy:", acc_5)
    print("Confusion Matrix:\n", cm_5)

    k_values = range(1, 101)
    accuracies = []
    for k in k_values:
        _, acc, _ = knn_classify(X_train_knn, y_train_knn, X_test_knn, y_test_knn, k)
        accuracies.append(acc)

    best_k = k_values[np.argmax(accuracies)]
    best_acc = max(accuracies)
    print("\n===== K-SWEEP RESULTS =====")
    print("Best K:", best_k)
    print("Best Accuracy:", best_acc)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker="o")
    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs K")
    plt.grid(True)
    plt.show()
```
