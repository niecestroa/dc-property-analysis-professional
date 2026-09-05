# Machine Learning Repository

Modular machine learning library with parallel R and Python implementations for classification, regression, and unsupervised methods. Includes clean workflows, reproducible pipelines, and model‑specific subfolders for clear organization and cross‑language comparison.

---

## Folder Structure

```
machine-learning/
    regression/
        pipelines/
        manual
```

Each folder contains model‑specific subdirectories. Every model includes:

- R implementation (`.R`)
- Python implementation (`.py`)

---

# Regression

Supervised learning methods for predicting continuous outcomes.

## Regression Models

| Model | R Library | R Function | Python Library | Python Function | When to Use | Why to Use |
|-------|-----------|------------|----------------|------------------|-------------|------------|
| Linear Regression | stats | lm() | sklearn.linear_model | LinearRegression() | When relationships are linear and interpretability is needed | Simple, transparent, and statistically grounded |
| Ridge | glmnet | glmnet(alpha=0) | sklearn.linear_model | Ridge() | When predictors are correlated or model overfits | Shrinks coefficients to reduce variance and improve stability |
| Lasso | glmnet | glmnet(alpha=1) | sklearn.linear_model | Lasso() | When feature selection is needed | Drives coefficients to zero, producing sparse models |
| Elastic Net | glmnet | glmnet(alpha=0.5) | sklearn.linear_model | ElasticNet() | When predictors are correlated and selection + shrinkage are needed | Balances Ridge and Lasso for more stable selection |
| PCR | pls | pcr() | sklearn.decomposition + sklearn.linear_model | PCA() + LinearRegression() | When multicollinearity is severe | Regresses on principal components to stabilize estimates |
| PLS | pls | plsr() | sklearn.cross_decomposition | PLSRegression() | When predictors are correlated and linked to outcome | Extracts components that maximize covariance with Y |
| Tree-Based Regression | rpart | rpart() | sklearn.tree | DecisionTreeRegressor() | When relationships are nonlinear or involve interactions | Captures nonlinearities without feature engineering |

---

# Design Philosophy

This repository emphasizes:

- **Modularity** — each model is self‑contained  
- **Reproducibility** — consistent workflows across R and Python  
- **Clarity** — minimal folder depth, intuitive organization  
- **Comparability** — parallel implementations for cross‑language benchmarking  

