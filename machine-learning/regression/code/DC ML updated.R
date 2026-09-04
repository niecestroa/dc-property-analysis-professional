############################################################
# Stat 627 Project – Modern Tidymodels Pipeline
# Author: Aaron Niecestro
# Original note: "Everything below takes too long to run. Also if one uses a laptop will run out of iterations..."
############################################################

#-----------------------------
# 0. Packages
#-----------------------------
library(tidyverse)
library(tidymodels)
library(janitor)

library(glmnet)    # for penalized regression
library(ranger)    # fast random forest
library(discrim)   # LDA/QDA via tidymodels
library(kernlab)   # SVM engine
library(plsmod)    # PCR/PLS via tidymodels
library(finetune)  # advanced tuning

tidymodels_prefer()

#-----------------------------
# 1. Load Raw Data
#-----------------------------
DC_Properties <- read_csv(
  "~/Documents/STAT 627 Statistical Machine Learning/Stat 627 Project/Data/DC_Properties.csv"
)

#-----------------------------
# 2. Data Cleaning (tidyverse)
#-----------------------------
DC_Final <- DC_Properties %>%
  # Replace all NA with 0 (as in original project)
  mutate(across(everything(), ~replace_na(., 0))) %>%
  
  # Select relevant variables (same as original)
  select(
    PRICE, BATHRM, HF_BATHRM, HEAT, AC, ROOMS, BEDRM, AYB, YR_RMDL, EYB,
    STORIES, QUALIFIED, GRADE, CNDTN, KITCHENS, FIREPLACES, WARD, QUADRANT,
    LATITUDE, LONGITUDE
  ) %>%
  
  # Convert numeric columns
  mutate(across(
    c(PRICE, BATHRM, HF_BATHRM, ROOMS, BEDRM, AYB, EYB, STORIES,
      KITCHENS, FIREPLACES, LATITUDE, LONGITUDE),
    as.numeric
  )) %>%
  
  # Convert categorical columns to character for cleaning
  mutate(
    HEAT      = as.character(HEAT),
    AC        = as.character(AC),
    QUALIFIED = as.character(QUALIFIED),
    GRADE     = as.character(GRADE),
    CNDTN     = as.character(CNDTN)
  ) %>%
  
  # Filter invalid / noisy rows (same logic as original)
  filter(
    CNDTN != "", CNDTN != "Default", CNDTN != "Poor",
    GRADE != "", GRADE != " No Data",
    HEAT != "No Data",
    PRICE > 10000, PRICE < 10000000,
    FIREPLACES < 10,
    KITCHENS <= 10,
    ROOMS <= 40,
    BEDRM <= 20,
    STORIES <= 10,
    LATITUDE != 0,
    LONGITUDE != 0
  ) %>%
  
  # Fix GRADE levels (collapse Exceptional-* to Exceptional)
  mutate(
    GRADE = case_when(
      str_detect(GRADE, "^Exceptional-") ~ "Exceptional",
      TRUE ~ GRADE
    )
  ) %>%
  
  # Create QUALIFIED_2 (binary factor)
  mutate(
    QUALIFIED_2 = factor(if_else(QUALIFIED == "Q", 1, 0))
  ) %>%
  
  # Fix AC coding
  mutate(
    AC = if_else(AC == "0", "N", AC)
  ) %>%
  
  # Create age variables (AYB.age, EYB.age, REMODEL.age)
  mutate(
    AYB.age = if_else(AYB == 2019, 0, 2019 - AYB),
    EYB.age = if_else(EYB == 2019, 0, 2019 - EYB),
    REMODEL.age = YR_RMDL %>%
      as.numeric() %>%
      replace_na(0) %>%
      {if_else(. %in% c(0, 20), 0, .)} %>%
      {2019 - .} %>%
      {if_else(. == 2019, 0, .)}
  ) %>%
  
  # Convert categorical variables to factors
  mutate(
    GRADE     = factor(GRADE),
    HEAT      = factor(HEAT),
    AC        = factor(AC),
    WARD      = factor(WARD),
    CONDITION = factor(CNDTN),
    QUALIFIED = factor(QUALIFIED)
  ) %>%
  
  # Final selection and derived variables
  mutate(
    PRICE_10K = PRICE / 10000,
    BATHRM    = BATHRM + HF_BATHRM * 0.5
  ) %>%
  select(
    PRICE, PRICE_10K, BATHRM, ROOMS, BEDRM, STORIES, KITCHENS, FIREPLACES,
    LATITUDE, LONGITUDE, AYB.age, EYB.age, REMODEL.age,
    HEAT, AC, QUALIFIED, QUALIFIED_2, GRADE, WARD,
    QUADRANT, CONDITION
  ) %>%
  
  # Remove impossible ages
  filter(AYB.age < 2000) %>%
  
  # Clean names just in case
  clean_names()

#-----------------------------
# 3. Train/Test Split (rsample)
#-----------------------------
set.seed(10000000)

data_split <- initial_split(DC_Final, prop = 0.75)
train_data <- training(data_split)
test_data  <- testing(data_split)

#-----------------------------
# 4. Recipes – Preprocessing
#-----------------------------
# Core regression recipe (polynomials + interactions + dummies + scaling)
price_recipe <- recipe(price_10k ~ ., data = train_data) %>%
  # Polynomial terms similar to original final_model.all
  step_poly(bathrm, degree = 8) %>%
  step_poly(rooms, degree = 1) %>%
  step_poly(bedrm, degree = 8) %>%
  step_poly(kitchens, degree = 1) %>%
  step_poly(fireplaces, degree = 4) %>%
  step_poly(ayb_age, eyb_age, remodel_age, degree = 4) %>%
  
  # Interaction CONDITION * BATHRM
  step_interact(~ condition:bathrm) %>%
  
  # Dummy encode categorical predictors
  step_dummy(all_nominal_predictors()) %>%
  
  # Remove zero-variance predictors
  step_zv(all_predictors()) %>%
  
  # Normalize numeric predictors
  step_normalize(all_numeric_predictors())

# Classification recipe for price categories (e.g., Reasonable vs Expensive)
train_data <- train_data %>%
  mutate(
    residential_class = if_else(price_10k < 57.42, "Reasonable", "Expensive") %>%
      factor()
  )

class_recipe <- recipe(residential_class ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

#-----------------------------
# 5. Model Specs (parsnip)
#-----------------------------

## 5.1 Linear Regression (baseline)
lm_spec <- linear_reg() %>%
  set_engine("lm")

## 5.2 Ridge Regression (mixture = 0)
ridge_spec <- linear_reg(
  penalty = tune(),
  mixture = 0
) %>%
  set_engine("glmnet")

## 5.3 LASSO Regression (mixture = 1)
lasso_spec <- linear_reg(
  penalty = tune(),
  mixture = 1
) %>%
  set_engine("glmnet")

## 5.4 Elastic Net (penalty + mixture tuned)
enet_spec <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

## 5.5 Random Forest (regression)
rf_spec <- rand_forest(
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## 5.6 PCR (Principal Components Regression)
pcr_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

pcr_recipe <- price_recipe %>%
  step_pca(all_predictors(), num_comp = tune())

## 5.7 PLS (Partial Least Squares)
pls_spec <- linear_reg() %>%
  set_engine("plsmod") %>%
  set_mode("regression")

pls_recipe <- price_recipe %>%
  step_pls(all_predictors(), outcome = "price_10k", num_comp = tune())

## 5.8 LDA / QDA (classification)
lda_spec <- discrim_linear() %>%
  set_engine("MASS") %>%
  set_mode("classification")

qda_spec <- discrim_quad() %>%
  set_engine("MASS") %>%
  set_mode("classification")

## 5.9 SVM (RBF kernel for classification)
svm_rbf_spec <- svm_rbf(
  cost      = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

#-----------------------------
# 6. Workflows
#-----------------------------

## 6.1 Linear Regression Workflow
lm_wflow <- workflow() %>%
  add_model(lm_spec) %>%
  add_recipe(price_recipe)

## 6.2 Ridge Workflow
ridge_wflow <- workflow() %>%
  add_model(ridge_spec) %>%
  add_recipe(price_recipe)

## 6.3 LASSO Workflow
lasso_wflow <- workflow() %>%
  add_model(lasso_spec) %>%
  add_recipe(price_recipe)

## 6.4 Elastic Net Workflow
enet_wflow <- workflow() %>%
  add_model(enet_spec) %>%
  add_recipe(price_recipe)

## 6.5 Random Forest Workflow
rf_wflow <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(price_recipe)

## 6.6 PCR Workflow
pcr_wflow <- workflow() %>%
  add_model(pcr_spec) %>%
  add_recipe(pcr_recipe)

## 6.7 PLS Workflow
pls_wflow <- workflow() %>%
  add_model(pls_spec) %>%
  add_recipe(pls_recipe)

## 6.8 LDA Workflow
lda_wflow <- workflow() %>%
  add_model(lda_spec) %>%
  add_recipe(class_recipe)

## 6.9 QDA Workflow
qda_wflow <- workflow() %>%
  add_model(qda_spec) %>%
  add_recipe(class_recipe)

## 6.10 SVM RBF Workflow
svm_wflow <- workflow() %>%
  add_model(svm_rbf_spec) %>%
  add_recipe(class_recipe)

#-----------------------------
# 7. Resampling (Cross-Validation)
#-----------------------------
set.seed(123)
cv_folds_reg <- vfold_cv(train_data, v = 5)
cv_folds_cls <- vfold_cv(train_data, v = 5)

#-----------------------------
# 8. Tuning Grids
#-----------------------------

## 8.1 Ridge / LASSO / ENet grid
lambda_grid <- grid_regular(
  penalty(range = c(-4, 1)), levels = 20
)

enet_grid <- grid_regular(
  penalty(range = c(-4, 1)),
  mixture(range = c(0, 1)),
  levels = 10
)

## 8.2 Random Forest grid
rf_grid <- grid_regular(
  mtry(range = c(3, 10)),
  trees(range = c(100, 500)),
  min_n(range = c(5, 25)),
  levels = 5
)

## 8.3 PCR / PLS components grid
comp_grid <- tibble(num_comp = 1:10)

## 8.4 SVM grid
svm_grid <- grid_regular(
  cost(range = c(-3, 3)),
  rbf_sigma(range = c(-3, 0)),
  levels = 7
)

#-----------------------------
# 9. Tuning – Examples
#-----------------------------

## 9.1 Ridge tuning
ridge_tuned <- tune_grid(
  ridge_wflow,
  resamples = cv_folds_reg,
  grid      = lambda_grid,
  metrics   = metric_set(rmse, rsq)
)

best_ridge <- select_best(ridge_tuned, "rmse")

ridge_final <- finalize_workflow(ridge_wflow, best_ridge) %>%
  fit(data = train_data)

## 9.2 Random Forest tuning
rf_tuned <- tune_grid(
  rf_wflow,
  resamples = cv_folds_reg,
  grid      = rf_grid,
  metrics   = metric_set(rmse, rsq)
)

best_rf <- select_best(rf_tuned, "rmse")

rf_final <- finalize_workflow(rf_wflow, best_rf) %>%
  fit(data = train_data)

## 9.3 SVM RBF tuning (classification)
svm_tuned <- tune_grid(
  svm_wflow,
  resamples = cv_folds_cls,
  grid      = svm_grid,
  metrics   = metric_set(accuracy)
)

best_svm <- select_best(svm_tuned, "accuracy")

svm_final <- finalize_workflow(svm_wflow, best_svm) %>%
  fit(data = train_data)

#-----------------------------
# 10. Evaluation on Test Set
#-----------------------------

## 10.1 Regression – Ridge
ridge_preds <- predict(ridge_final, test_data) %>%
  bind_cols(test_data)

ridge_metrics <- ridge_preds %>%
  metrics(truth = price_10k, estimate = .pred)

## 10.2 Regression – Random Forest
rf_preds <- predict(rf_final, test_data) %>%
  bind_cols(test_data)

rf_metrics <- rf_preds %>%
  metrics(truth = price_10k, estimate = .pred)

## 10.3 Classification – SVM
test_data_cls <- test_data %>%
  mutate(
    residential_class = if_else(price_10k < 57.42, "Reasonable", "Expensive") %>%
      factor()
  )

svm_preds <- predict(svm_final, test_data_cls) %>%
  bind_cols(test_data_cls)

svm_metrics <- svm_preds %>%
  metrics(truth = residential_class, estimate = .pred_class)

#-----------------------------
# 11. Variable Importance (Random Forest)
#-----------------------------
rf_fit <- extract_fit_parsnip(rf_final)

rf_importance <- ranger::importance(rf_fit$fit)
