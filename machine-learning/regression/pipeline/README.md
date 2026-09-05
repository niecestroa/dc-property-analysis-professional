# **DC Property ML — Regression Models**

This folder contains the regression components of the **DC Property ML** project, a full machine‑learning pipeline for modeling Washington, D.C. real estate prices. The codebase includes reproducible R workflows (and optional Python equivalents) for predicting continuous outcomes such as `Price_10K`.

The regression module is part of a larger system that includes preprocessing, feature engineering, model training, evaluation, caching, and visualization.

## **Included Regression Methods**

The following supervised learning models are implemented:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **PCR** — Principal Components Regression  
- **PLS** — Partial Least Squares  
- **Regression Trees**
- **Bagging**
- **Random Forest Regression**

Each model follows a consistent interface and integrates with the project’s unified evaluation and caching system.

## **Folder Structure**

Each regression model is organized into its own subfolder:

```
regression/
│
├── linear/
├── ridge/
├── lasso/
├── pcr/
├── pls/
├── tree/
├── bagging/
└── random_forest/
```

Every subfolder contains:

- **R implementation** (`.R`)
- **Optional Python implementation** (`.py`)
- Supporting utilities (plots, metrics, cached artifacts)

This modular layout keeps the codebase clean, navigable, and aligned with the project’s end‑to‑end pipeline.

## **Pipeline Integration**

All regression models plug into the main workflow:

- Preprocessing and feature engineering  
- Train/test split  
- Model training (with caching)  
- Unified evaluation (MSE, RMSE, MAE, R²)  
- Visualization (predicted vs actual, MSE comparison charts)  
- Exported comparison tables  

The regression module is fully compatible with:

- `run_all.R`  
- `config.yaml`  
- The caching system (`cache/`)  
- The evaluation framework (`14_evaluate.R`)  

## **Purpose**

This folder highlights the regression modeling techniques used in the DC Property ML project while keeping the codebase:

- **modular**  
- **reproducible**  
- **easy to extend**  
- **easy to compare across models**  

It serves as the foundation for the project’s continuous‑outcome prediction tasks.
parison** section  

Just tell me which one you want next.
