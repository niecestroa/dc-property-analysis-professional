# =============================================================================
# 14_evaluate.R
# Unified evaluation functions for regression and classification models.
# Produces the model comparison table from the paper and saves all metrics.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

# ===========================================================================
# Regression Metrics
# ===========================================================================

#' Compute standard regression metrics for one model
#'
#' @param actual  Numeric vector of true values.
#' @param pred    Numeric vector of predicted values.
#' @param label   Model name string.
#' @return Named numeric vector: MSE, RMSE, MAE, R2.
regression_metrics <- function(actual, pred, label = "model") {
  residuals <- actual - pred
  mse  <- mean(residuals^2)
  rmse <- sqrt(mse)
  mae  <- mean(abs(residuals))
  ss_res <- sum(residuals^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r2   <- 1 - ss_res / ss_tot

  metrics <- c(MSE = mse, RMSE = rmse, MAE = mae, R2 = r2)
  message(sprintf("[regression_metrics] %s → MSE=%.4f | RMSE=%.4f | MAE=%.4f | R2=%.4f",
                  label, mse, rmse, mae, r2))
  metrics
}

#' Build the full regression comparison table across all models
#'
#' @param results_list Named list of lists, each with $pred and $actual.
#' @return Tibble sorted by MSE ascending.
regression_comparison_table <- function(results_list) {
  purrr::imap_dfr(results_list, function(res, model_name) {
    m <- regression_metrics(res$actual, res$pred, label = model_name)
    tibble::tibble(
      Model = model_name,
      MSE   = m["MSE"],
      RMSE  = m["RMSE"],
      MAE   = m["MAE"],
      R2    = m["R2"]
    )
  }) %>%
    dplyr::arrange(MSE)
}

# ===========================================================================
# Classification Metrics
# ===========================================================================

#' Compute classification metrics for one model
#'
#' @param actual  Factor vector of true classes.
#' @param pred    Factor vector of predicted classes.
#' @param label   Model name string.
#' @return Named numeric vector: Accuracy, Error_Rate, plus per-class precision/recall.
classification_metrics <- function(actual, pred, label = "model") {
  pred   <- factor(pred,   levels = levels(actual))
  conf   <- table(Predicted = pred, Actual = actual)
  acc    <- sum(diag(conf)) / sum(conf)
  err    <- 1 - acc

  message(sprintf("[classification_metrics] %s → Accuracy=%.6f | Error=%.6f",
                  label, acc, err))
  print(conf)
  c(Accuracy = acc, Error_Rate = err)
}

#' Build the classification comparison table
#'
#' @param results_list Named list of lists, each with $pred and $actual (factor).
#' @return Tibble sorted by Accuracy descending.
classification_comparison_table <- function(results_list) {
  purrr::imap_dfr(results_list, function(res, model_name) {
    m <- classification_metrics(res$actual, res$pred, label = model_name)
    tibble::tibble(
      Model      = model_name,
      Accuracy   = m["Accuracy"],
      Error_Rate = m["Error_Rate"]
    )
  }) %>%
    dplyr::arrange(dplyr::desc(Accuracy))
}

# ===========================================================================
# Cross-Validation MSE (Validation-Set Approach)
# ===========================================================================

#' Estimate test MSE using k-fold cross-validation on an lm formula
#'
#' @param df      Full dataset (train + test combined, or just train).
#' @param formula Formula for lm().
#' @param k       Number of folds. Default 10.
#' @param seed    Random seed.
#' @return Scalar CV MSE.
cv_mse_lm <- function(df, formula, k = 10, seed = 627) {
  set.seed(seed)
  n      <- nrow(df)
  folds  <- sample(rep(1:k, length.out = n))
  errors <- numeric(k)

  for (i in seq_len(k)) {
    fold_train <- df[folds != i, ]
    fold_test  <- df[folds == i, ]
    fit  <- lm(formula, data = fold_train)
    pred <- predict(fit, newdata = fold_test)
    errors[i] <- mean((fold_test$Price_10K - pred)^2)
  }

  cv_mse <- mean(errors)
  message(sprintf("[cv_mse_lm] %d-fold CV MSE = %.4f", k, cv_mse))
  cv_mse
}

# ===========================================================================
# Save outputs
# ===========================================================================

#' Save a comparison table as CSV
#'
#' @param tbl     Tibble from regression_comparison_table() or classification_comparison_table().
#' @param cfg     Config list.
#' @param tag     File tag, e.g. "regression_comparison".
save_comparison_table <- function(tbl, cfg, tag = "model_comparison") {
  out <- file.path(here::here(cfg$paths$table_dir), paste0(tag, ".csv"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(tbl, out)
  message(sprintf("[save_comparison_table] Saved → %s", out))
}

#' Plot predicted vs actual for a regression model
#'
#' @param actual  Numeric vector.
#' @param pred    Numeric vector.
#' @param cfg     Config list.
#' @param label   Model name (used in title and filename).
save_pred_vs_actual <- function(actual, pred, cfg, label = "model") {
  out <- file.path(here::here(cfg$paths$plot_dir),
                   paste0("pred_vs_actual_", gsub(" ", "_", label), ".png"))

  df_plot <- data.frame(Actual = actual, Predicted = pred)
  lim     <- range(c(actual, pred))

  p <- ggplot2::ggplot(df_plot, ggplot2::aes(x = Actual, y = Predicted)) +
    ggplot2::geom_point(alpha = 0.3, size = 0.7, colour = "steelblue") +
    ggplot2::geom_abline(slope = 1, intercept = 0,
                         colour = "firebrick", linewidth = 0.8, linetype = "dashed") +
    ggplot2::coord_equal(xlim = lim, ylim = lim) +
    ggplot2::labs(
      title = paste(label, "— Predicted vs Actual"),
      x = "Actual Price_10K", y = "Predicted Price_10K"
    ) +
    ggplot2::theme_bw()

  ggplot2::ggsave(out, p, width = 7, height = 7, dpi = 150)
  message(sprintf("[save_pred_vs_actual] Saved → %s", out))
  invisible(p)
}

#' Bar chart comparing model MSEs side-by-side
#'
#' @param comparison_tbl Tibble from regression_comparison_table().
#' @param cfg            Config list.
save_mse_bar_chart <- function(comparison_tbl, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir), "model_mse_comparison.png")

  p <- ggplot2::ggplot(comparison_tbl,
                       ggplot2::aes(x = reorder(Model, MSE), y = MSE, fill = MSE)) +
    ggplot2::geom_col(show.legend = FALSE) +
    ggplot2::scale_fill_gradient(low = "steelblue", high = "firebrick") +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "Model Comparison — Test MSE",
                  x = NULL, y = "Test MSE") +
    ggplot2::theme_bw()

  ggplot2::ggsave(out, p, width = 9, height = 5, dpi = 150)
  message(sprintf("[save_mse_bar_chart] Saved → %s", out))
  invisible(p)
}
