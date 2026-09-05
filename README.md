# **DC Property Modeling Lab**

## **Overview**
The **DC Property Modeling Lab** is a unified analytical framework integrating statistical modeling, Bayesian inference, generalized linear models, and modern machine‑learning workflows. The project provides a complete, reproducible system for modeling Washington, D.C. real estate data using professional‑grade analytical engineering practices.

This lab combines:

- Classical statistical modeling (GLM, inference, diagnostics)  
- Bayesian hierarchical modeling (Stan, JAGS)  
- Machine learning pipelines (trees, ensembles, SVM, regularization)  
- Modular R workflows with optional Python equivalents  
- Automated evaluation, visualization, and reporting  
- A structured ML manual documenting methods and pipeline design  

The goal is to demonstrate rigorous modeling, reproducible computation, and clear analytical communication in a single cohesive system.

---

## **Key Features**

### **1. End‑to‑End ML Pipeline**
A complete workflow for DC property modeling:

- Data cleaning and preprocessing  
- Feature engineering  
- Train/test splitting  
- Regression and classification models  
- Ensemble methods (bagging, random forest)  
- Unified evaluation (MSE, RMSE, MAE, R², accuracy)  
- Automated plots and comparison tables  
- Caching for fast reproducible runs  

All pipeline components are controlled through a central `config.yaml`.

---

### **2. Bayesian Modeling**
Bayesian workflows using **Stan** and **JAGS**, including:

- Hierarchical model specification  
- Posterior inference and uncertainty quantification  
- MCMC diagnostics and convergence checks  
- Posterior predictive validation  
- Model comparison and interpretation  

These models provide probabilistic insight complementary to ML predictions.

---

### **3. Generalized Linear Models (GLM)**
Classical statistical modeling using:

- Linear regression  
- Logistic regression  
- Poisson and negative binomial models  
- Regularization (ridge, lasso, elastic net)  
- Cross‑validation and diagnostics  

GLMs serve as interpretable baselines within the modeling lab.

---

### **4. Machine Learning Manual**
A structured reference documenting:

- Modeling assumptions  
- Feature engineering strategies  
- Hyperparameter selection  
- Evaluation metrics  
- Pipeline architecture  
- Best practices for reproducible ML  

This manual supports transparency and professional communication.

---

## **Repository Structure**

```
dc-property-modeling-lab/
│
├── data/                 # Raw and processed datasets
├── r_code/               # Pipeline modules and model implementations
├── bayesian/             # Stan/JAGS models
├── glm/                  # GLM workflows
├── ml_manual/            # Modeling documentation
├── models/               # Saved model artifacts (cached)
├── outputs/              # Plots, tables, reports
└── config.yaml           # Central configuration file
```

---

## **Technical Capabilities Demonstrated**

| Area | Capabilities | Tools |
|------|--------------|-------|
| **Statistical Modeling** | GLMs, inference, diagnostics | R, Python |
| **Bayesian Analysis** | Hierarchical models, MCMC | Stan, JAGS |
| **Machine Learning** | Trees, ensembles, SVM, regularization | R, Python |
| **Pipeline Engineering** | Modular design, caching, reproducibility | R Projects, Git |
| **Visualization** | Exploratory and model‑based graphics | ggplot2, Python |
| **Documentation** | ML manual, structured reporting | Quarto, R Markdown |

---

## **Companion Repository**
For earlier academic case studies and foundational modeling work, see:

**DC Property Academic Statistical Analysis Portfolio**  
[https://github.com/niecestroa/dc-property-analysis-academic/tree/main](https://github.com/niecestroa/dc-property-analysis-academic/tree/main)

---
