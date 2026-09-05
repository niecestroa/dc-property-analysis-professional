# **DC Property Modeling Lab**

A unified analytical system for modeling Washington, D.C. real estate data using rigorous statistical methods, Bayesian inference, and modern machine‑learning engineering practices. The **DC Property Modeling Lab** provides a fully reproducible, modular workflow designed for professional‑grade modeling, diagnostics, and reporting.

This repository serves as a comprehensive modeling environment that integrates:

- Classical statistical modeling (GLM, inference, diagnostics)  
- Bayesian hierarchical modeling (Stan, JAGS)  
- Machine‑learning regression pipelines (trees, ensembles, regularization)  
- Modular R workflows with optional Python equivalents  
- Automated evaluation, visualization, and reporting  
- A structured ML manual documenting pipeline architecture and modeling methodology  

The goal is to demonstrate **end‑to‑end analytical engineering**, **transparent modeling**, and **reproducible computation** in a single cohesive system.

---

## **Key Features**

### **1. End‑to‑End Modeling Pipeline**
A complete, configurable workflow for DC property modeling:

- Data cleaning and preprocessing  
- Feature engineering  
- Train/test splitting  
- Regression models (linear, regularized, tree‑based, ensembles)  
- Unified evaluation metrics (MSE, RMSE, MAE, R²)  
- Automated plots, diagnostics, and comparison tables  
- Caching for fast reproducible runs  

All components are controlled through a central `config.yaml`, enabling consistent and repeatable analyses.

---

### **2. Bayesian Modeling**
Bayesian workflows implemented in **Stan** and **JAGS**, including:

- Hierarchical model specification  
- Posterior inference and uncertainty quantification  
- MCMC diagnostics and convergence checks  
- Posterior predictive validation  
- Model comparison and interpretation  

These models provide probabilistic insight complementary to classical and ML regression approaches.

---

### **3. Generalized Linear Models (GLM)**
Classical statistical modeling using:

- Linear regression  
- Regularization (ridge, lasso, elastic net)  
- Cross‑validation  
- Residual diagnostics and interpretability workflows  

GLMs serve as interpretable baselines and anchor the modeling lab’s statistical foundation.

---

### **4. Machine‑Learning Regression**
Modern ML regression pipelines including:

- Decision trees  
- Random forest  
- Gradient boosting (if enabled)  
- Regularized regression variants  
- Unified evaluation and visualization  

Classification has been removed from the current version of the lab to maintain methodological coherence with the dataset and modeling objectives.

---

### **5. Machine Learning Manual**
A structured reference documenting:

- Modeling assumptions  
- Feature engineering strategies  
- Hyperparameter selection  
- Evaluation metrics  
- Pipeline architecture  
- Best practices for reproducible ML  

This manual supports transparency, communication, and professional documentation of the modeling system.

---

## **Technical Capabilities Demonstrated**

| Area | Capabilities | Tools |
|------|--------------|-------|
| **Statistical Modeling** | GLMs, inference, diagnostics | R, Python |
| **Bayesian Analysis** | Hierarchical models, MCMC | Stan, JAGS |
| **Machine Learning (Regression)** | Trees, ensembles, regularization | R, Python |
| **Pipeline Engineering** | Modular design, caching, reproducibility | R Projects, Git |
| **Visualization** | Exploratory and model‑based graphics | ggplot2, Python |
| **Documentation** | ML manual, structured reporting | Quarto, R Markdown |

---

## **Companion Repository**

For earlier academic case studies and foundational modeling work, see:

**DC Property Academic Statistical Analysis Portfolio**  
[https://github.com/niecestroa/dc-property-analysis-academic/tree/main](https://github.com/niecestroa/dc-property-analysis-academic/tree/main)

---

## **Purpose and Vision**

The **DC Property Modeling Lab** is designed as a professional modeling environment that demonstrates:

- Reproducible analytical engineering  
- Transparent statistical and Bayesian modeling  
- Modern ML regression workflows  
- Clean documentation and communication of modeling decisions  

It serves as a showcase of rigorous modeling practice suitable for data science, biostatistics, and applied analytics roles.
