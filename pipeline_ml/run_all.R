# =============================================================================
# run_all.R — DC Property ML End-to-End Workflow
# =============================================================================
# Usage: Rscript run_all.R               (runs everything)
#        Rscript run_all.R --skip-data   (assumes data/splits already exist)
# =============================================================================

suppressPackageStartupMessages(library(here))

# ---- 0. Bootstrap: resolve project root via {here} -------------------------
cat("\n", strrep("=", 70), "\n")
cat("  DC Property ML — End-to-End Workflow\n")
cat("  Author: Aaron Niecestro\n")
cat(strrep("=", 70), "\n\n")

# ---- 1. Source all modules --------------------------------------------------
# IMPORTANT: Your project uses r_code/, NOT src/
source(here("r_code", "01_load_config.R"))
source(here("r_code", "02_load_data.R"))
source(here("r_code", "03_split_data.R"))
source(here("r_code", "04_feature_engineering.R"))
source(here("r_code", "05_models_linear.R"))
source(here("r_code", "06_models_polynomial.R"))
source(here("r_code", "07_models_knn.R"))
source(here("r_code", "08_models_lda_qda.R"))
source(here("r_code", "09_models_ridge_lasso.R"))
source(here("r_code", "10_models_pcr_pls.R"))
source(here("r_code", "11_models_trees.R"))
source(here("r_code", "12_models_random_forest.R"))
# For classification only, which ignored for this project
source(here("r_code", "13_models_svm.R"))
source(here("r_code", "regression_metrics.R")) # Need this for regression models

# utils folder is correct — keep these
source(here("utils", "utils_io.R"))
source(here("utils", "utils_plots.R"))
source(here("utils", "utils_preprocessing.R"))

# ---- 2. Load config ---------------------------------------------------------
cfg <- load_config()
set_project_seed(cfg)
ensure_output_dirs(cfg)

args <- commandArgs(trailingOnly = TRUE)
skip_data <- "--skip-data" %in% args

# ===========================================================================
# PHASE 1 — DATA
# ===========================================================================
cat("\n--- PHASE 1: Data Loading & Splitting ---\n")

if (!skip_data) {
  df    <- load_raw_data(cfg)
  save_processed_data(df, cfg)
  splits <- make_split(df, cfg)
} else {
  message("[run_all] --skip-data: loading pre-saved splits.")
  df     <- readRDS(here(cfg$paths$processed_data))
  splits <- load_splits(cfg)
}

train <- splits$train
test  <- splits$test

# Add classification labels (used by KNN / LDA / QDA / SVM)
labeled <- add_classification_labels(train, test, cfg)
train_c <- labeled$train
test_c  <- labeled$test

# ===========================================================================
# PHASE 2 — EDA
# ===========================================================================
cat("\n--- PHASE 2: Exploratory Data Analysis ---\n")

cat("\nData Quality Report:\n")
data_quality_report(df)

cat("\nDescriptive Statistics:\n")
print(numeric_summary(df))

# Save EDA plots
save_price_histogram(df, cfg)
save_geo_price_map(df, cfg)
save_correlation_heatmap(df, cfg)
for (cat_var in c("Grade", "Condition", "Ward")) {
  if (cat_var %in% colnames(df)) save_price_by_category(df, cat_var, cfg)
}

# ===========================================================================
# PHASE 3 — FEATURE ENGINEERING
# ===========================================================================
cat("\n--- PHASE 3: Feature Engineering ---\n")

train_poly <- add_polynomial_features(train, cfg)
test_poly  <- add_polynomial_features(test,  cfg)

# Build glmnet matrices (for Ridge / Lasso / PCR / PLS)
train_Xy <- build_Xy(train_poly)
test_Xy  <- build_Xy(test_poly)

# ===========================================================================
# PHASE 4 — REGRESSION MODELS
# ===========================================================================
cat("\n--- PHASE 4: Regression Models ---\n")

## 4a. Linear Regression
cat("\n[4a] Linear Regression\n")
lm_model <- fit_linear(train)
save_model(lm_model, cfg, "linear")
save_residual_plots(lm_model, cfg, tag = "linear_diagnostics")
check_vif(lm_model)

lm_pred   <- predict(lm_model, newdata = test)
lm_metrics <- regression_metrics(test$Price_10K, lm_pred, "Linear")
log_result(cfg, "Linear", "Test_MSE", lm_metrics["MSE"])
save_pred_vs_actual(test$Price_10K, lm_pred, cfg, "Linear Regression")

## 4b. Polynomial Regression
cat("\n[4b] Polynomial Regression\n")
poly_model    <- fit_polynomial(train, cfg)
outlier_idx   <- detect_outliers(poly_model)
poly_refitted <- refit_without_outliers(train, cfg, outlier_idx)
poly_model    <- poly_refitted$model
save_model(poly_model, cfg, "polynomial")
save_diagnostic_plots(poly_model, cfg, tag = "polynomial_diagnostics")

poly_pred    <- predict(poly_model, newdata = test)
poly_metrics <- regression_metrics(test$Price_10K, poly_pred, "Polynomial")
log_result(cfg, "Polynomial", "Test_MSE", poly_metrics["MSE"])
save_pred_vs_actual(test$Price_10K, poly_pred, cfg, "Polynomial Regression")

## 4c. Ridge Regression
cat("\n[4c] Ridge Regression\n")
ridge_model <- fit_ridge(train_Xy, cfg)
save_model(ridge_model, cfg, "ridge")
save_cv_plot(ridge_model, cfg, "ridge_cv")
save_coef_path_plot(ridge_model, cfg, "ridge_coef_path")

ridge_eval <- eval_ridge(ridge_model, test_Xy)
log_result(cfg, "Ridge", "Test_MSE", ridge_eval$mse)

## 4d. Lasso Regression
cat("\n[4d] Lasso Regression\n")
lasso_model <- fit_lasso(train_Xy, cfg)
save_model(lasso_model, cfg, "lasso")
save_cv_plot(lasso_model, cfg, "lasso_cv")
save_coef_path_plot(lasso_model, cfg, "lasso_coef_path")

lasso_eval <- eval_lasso(lasso_model, test_Xy)
log_result(cfg, "Lasso", "Test_MSE", lasso_eval$mse)
compare_ridge_lasso(ridge_eval, lasso_eval)

## 4e. PCR
cat("\n[4e] Principal Component Regression (PCR)\n")
pcr_model <- fit_pcr(train_poly, cfg)
save_model(pcr_model, cfg, "pcr")
save_rmsep_plot(pcr_model, cfg, "pcr_rmsep")
save_variance_plot(pcr_model, cfg, "pcr_variance")

pcr_eval    <- eval_pcr(pcr_model, test_poly)
log_result(cfg, "PCR", "Test_MSE", pcr_eval$mse,
           notes = paste("ncomp =", pcr_eval$ncomp_used))

## 4f. PLS
cat("\n[4f] Partial Least Squares (PLS)\n")
pls_model <- fit_pls(train_poly, cfg)
save_model(pls_model, cfg, "pls")
save_rmsep_plot(pls_model, cfg, "pls_rmsep")
save_variance_plot(pls_model, cfg, "pls_variance")

pls_eval    <- eval_pls(pls_model, test_poly)
log_result(cfg, "PLS", "Test_MSE", pls_eval$mse,
           notes = paste("ncomp =", pls_eval$ncomp_used))

## 4g. Regression Tree + Pruning
cat("\n[4g] Regression Tree (with pruning)\n")
tree_model   <- fit_tree(train, cfg)
save_tree_plot(tree_model, cfg, "tree_unpruned")
save_cp_plot(tree_model, cfg)

pruned_tree  <- prune_tree(tree_model, cfg)
save_tree_plot(pruned_tree, cfg, "tree_pruned")
save_model(pruned_tree, cfg, "tree_pruned")

tree_eval    <- eval_tree(pruned_tree, test)
log_result(cfg, "Regression Tree (Pruned)", "Test_MSE", tree_eval$mse)

## 4h. Random Forest + Bagging
cat("\n[4h] Random Forest & Bagging\n")
rf_model    <- fit_random_forest(train, cfg)
save_model(rf_model, cfg, "random_forest_full")
save_rf_error_plot(rf_model, cfg, "rf_oob_error_full")
save_importance_plot(rf_model, cfg, "rf_importance_full")

rf_optimal  <- fit_optimal_rf(train, cfg)
save_model(rf_optimal, cfg, "random_forest_optimal")
save_importance_plot(rf_optimal, cfg, "rf_importance_optimal")

bag_model   <- fit_bagging(train, cfg)
save_model(bag_model, cfg, "bagging")

rf_eval     <- eval_rf(rf_optimal, test)
log_result(cfg, "Random Forest (Optimal)", "Test_MSE", rf_eval$mse)
save_pred_vs_actual(test$Price_10K, rf_eval$pred, cfg, "Random Forest Optimal")

# ===========================================================================
# PHASE 5 — CLASSIFICATION MODELS
# ===========================================================================
cat("\n--- PHASE 5: Classification Models ---\n")

# ============================================================
# 5A — BINARY CLASSIFICATION (mean / median)
# ============================================================

cat("\n[5a] Binary Classification (Mean / Median)\n")

## KNN (binary)
knn_results <- knn_loop(train_c, test_c, cfg = cfg, class_col = "class_mean")
plot_knn_error(knn_results, cfg)
best_k      <- knn_results$k[which.min(knn_results$mse)]
knn_best    <- run_knn(train_c, test_c, k = best_k, class_col = "class_mean")

## LDA/QDA (binary)
lda_model <- fit_lda(train_c, "class_mean", priors = cfg$models$lda$priors_equal)
lda_eval  <- eval_lda(lda_model, test_c, "class_mean")

qda_model <- fit_qda(train_c, "class_mean", priors = cfg$models$qda$priors_equal)
qda_eval  <- eval_qda(qda_model, test_c, "class_mean")

## SVM (binary)
save_svm_scatter(test_c, "class_median", cfg)
svm_kernel_results <- compare_svm_kernels(train_c, test_c, "class_median", cfg)
best_kernel <- svm_kernel_results$Kernel[which.min(svm_kernel_results$Misclassification_Rate)]
svm_best    <- fit_svm(train_c, kernel = best_kernel, class_col = "class_median", cfg = cfg)
svm_eval    <- eval_svm(svm_best, test_c, "class_median")


# ============================================================
# 5B — 3-CLASS CLASSIFICATION (Low / Medium / High)
# ============================================================

cat("\n[5b-2] 3-Class Classification (Low / Medium / High)\n")

## KNN (3-class)
knn_results_3 <- knn_loop(train_c, test_c, cfg = cfg, class_col = "class_price3")
best_k_3      <- knn_results_3$k[which.min(knn_results_3$mse)]
knn_best_3    <- run_knn(train_c, test_c, k = best_k_3, class_col = "class_price3")

## LDA/QDA (3-class)
lda_model_3 <- fit_lda(train_c, "class_price3", priors = cfg$models$lda$priors_equal)
lda_eval_3  <- eval_lda(lda_model_3, test_c, "class_price3")

qda_model_3 <- fit_qda(train_c, "class_price3", priors = cfg$models$qda$priors_equal)
qda_eval_3  <- eval_qda(qda_model_3, test_c, "class_price3")

## SVM (3-class)
svm_kernel_results_3 <- compare_svm_kernels(train_c, test_c, "class_price3", cfg)
best_kernel_3        <- svm_kernel_results_3$Kernel[which.min(svm_kernel_results_3$Misclassification_Rate)]
svm_best_3           <- fit_svm(train_c, kernel = best_kernel_3, class_col = "class_price3", cfg = cfg)
svm_eval_3           <- eval_svm(svm_best_3, test_c, "class_price3")


# ============================================================
# 5C — 4-CLASS CLASSIFICATION (Quartiles)
# ============================================================

cat("\n[5b-3] 4-Class Classification (Quartiles)\n")

## KNN (4-class)
knn_results_4 <- knn_loop(train_c, test_c, cfg = cfg, class_col = "class_price4")
best_k_4      <- knn_results_4$k[which.min(knn_results_4$mse)]
knn_best_4    <- run_knn(train_c, test_c, k = best_k_4, class_col = "class_price4")

## LDA/QDA (4-class)
lda_model_4 <- fit_lda(train_c, "class_price4", priors = cfg$models$lda$priors_equal)
lda_eval_4  <- eval_lda(lda_model_4, test_c, "class_price4")

qda_model_4 <- fit_qda(train_c, "class_price4", priors = cfg$models$qda$priors_equal)
qda_eval_4  <- eval_qda(qda_model_4, test_c, "class_price4")

## SVM (4-class)
svm_kernel_results_4 <- compare_svm_kernels(train_c, test_c, "class_price4", cfg)
best_kernel_4        <- svm_kernel_results_4$Kernel[which.min(svm_kernel_results_4$Misclassification_Rate)]
svm_best_4           <- fit_svm(train_c, kernel = best_kernel_4, class_col = "class_price4", cfg = cfg)
svm_eval_4           <- eval_svm(svm_best_4, test_c, "class_price4")


# ===========================================================================
# PHASE 6 — MODEL COMPARISON REPORT
# ===========================================================================
cat("\n--- PHASE 6: Model Comparison Summary ---\n")

# ============================================================
# Regression comparison
# ============================================================

reg_results <- list(
  "Linear Regression"          = list(actual = test$Price_10K, pred = lm_pred),
  "Polynomial Regression"      = list(actual = test$Price_10K, pred = poly_pred),
  "PCR"                        = list(actual = test$Price_10K, pred = pcr_eval$pred),
  "PLS"                        = list(actual = test$Price_10K, pred = pls_eval$pred),
  "Regression Tree (Pruned)"   = list(actual = test$Price_10K, pred = tree_eval$pred),
  "Random Forest (Optimal)"    = list(actual = test$Price_10K, pred = rf_eval$pred)
)
reg_table <- regression_comparison_table(reg_results)
save_comparison_table(reg_table, cfg, "regression_comparison")
save_mse_bar_chart(reg_table, cfg)

saveRDS(reg_table, here(cfg$paths$model_dir, "regression_comparison_table.rds"))

# ============================================================
# Binary classification comparison
# ============================================================

cls_results <- list(
  "KNN (Best k)"               = list(actual = test_c$class_mean, pred = knn_best$pred),
  "LDA (equal priors)"         = list(actual = test_c$class_mean, pred = lda_eval$pred),
  "QDA (equal priors)"         = list(actual = test_c$class_mean, pred = qda_eval$pred),
  "SVM (Best Kernel)"          = list(actual = test_c$class_median, pred = svm_eval$pred)
)
cls_table <- classification_comparison_table(cls_results)
save_comparison_table(cls_table, cfg, "classification_comparison")
save_classification_bar_chart(cls_table, cfg, "classification_binary")

saveRDS(cls_table, here(cfg$paths$model_dir, "classification_binary_table.rds"))

# ============================================================
# 3-class comparison
# ============================================================

cls_results_3 <- list(
  "KNN (Best k)" = list(actual = test_c$class_price3, pred = knn_best_3$pred),
  "LDA"          = list(actual = test_c$class_price3, pred = lda_eval_3$pred),
  "QDA"          = list(actual = test_c$class_price3, pred = qda_eval_3$pred),
  "SVM"          = list(actual = test_c$class_price3, pred = svm_eval_3$pred)
)
cls_table_3 <- classification_comparison_table(cls_results_3)
save_comparison_table(cls_table_3, cfg, "classification_comparison_3class")
save_classification_bar_chart(cls_table_3, cfg, "classification_3class")

saveRDS(cls_table_3, here(cfg$paths$model_dir, "classification_3class_table.rds"))

# ============================================================
# 4-class comparison
# ============================================================

cls_results_4 <- list(
  "KNN (Best k)" = list(actual = test_c$class_price4, pred = knn_best_4$pred),
  "LDA"          = list(actual = test_c$class_price4, pred = lda_eval_4$pred),
  "QDA"          = list(actual = test_c$class_price4, pred = qda_eval_4$pred),
  "SVM"          = list(actual = test_c$class_price4, pred = svm_eval_4$pred)
)
cls_table_4 <- classification_comparison_table(cls_results_4)
save_comparison_table(cls_table_4, cfg, "classification_comparison_4class")
save_classification_bar_chart(cls_table_4, cfg, "classification_4class")

saveRDS(cls_table_4, here(cfg$paths$model_dir, "classification_4class_table.rds"))

# ============================================================
# Confusion Matrix
# ============================================================

save_confusion_matrix <- function(actual, pred, cfg, tag) {
  cm <- table(Predicted = pred, Actual = actual)
  out <- file.path(here::here(cfg$paths$plot_dir), paste0("confusion_", tag, ".png"))
  png(out, width = 800, height = 600)
  print(cm)
  dev.off()
  message("[save_confusion_matrix] Saved → ", out)
}

# Binary
save_confusion_matrix(test_c$class_mean, knn_best$pred, cfg, "binary_knn")

# 3-class
save_confusion_matrix(test_c$class_price3, knn_best_3$pred, cfg, "3class_knn")

# 4-class
save_confusion_matrix(test_c$class_price4, knn_best_4$pred, cfg, "4class_knn")

# ============================================================
# Final unified comparison
# ============================================================

cat("\n", strrep("=", 70), "\n")
cat("  FINAL COMPARISON: Regression vs Binary vs 3-Class vs 4-Class\n")
cat(strrep("=", 70), "\n")

cat("\n--- Regression Models ---\n")
print(reg_table)

cat("\n--- Binary Classification ---\n")
print(cls_table)

cat("\n--- 3-Class Classification ---\n")
print(cls_table_3)

cat("\n--- 4-Class Classification ---\n")
print(cls_table_4)

cat("\n", strrep("=", 70), "\n")
cat("  Run complete. All outputs in outputs/\n")
cat(strrep("=", 70), "\n\n")
