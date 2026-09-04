# =============================================================================
# 10_models_pcr_pls.R
# Principal Component Regression (PCR) and Partial Least Squares (PLS).
# Uses the pls package. Both models applied to the polynomial feature matrix.
# Paper findings: PCR best at 29 components; PLS RMSEP stabilises at ~3 comps.
# =============================================================================

suppressPackageStartupMessages({
  library(pls)
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# PCR
# ---------------------------------------------------------------------------

#' Fit PCR model with cross-validation
#'
#' @param train   Training tibble.
#' @param cfg     Config list.
#' @param formula Model formula. Defaults to polynomial response ~ all numerics.
#' @return Fitted mvr (PCR) object.
fit_pcr <- function(train, cfg, formula = NULL) {
  ncomp_max  <- cfg$models$pcr$ncomp_max     # 31
  validation <- cfg$models$pcr$validation    # "CV"

  if (is.null(formula)) {
    num_preds <- train %>%
      dplyr::select(where(is.numeric), -Price_10K, -dplyr::starts_with("class_")) %>%
      colnames()
    formula <- as.formula(paste("Price_10K ~", paste(num_preds, collapse = " + ")))
  }

  message(sprintf("[fit_pcr] Fitting PCR (ncomp=%d, validation=%s) ...",
                  ncomp_max, validation))
  model <- pls::pcr(formula,
                    data       = train,
                    ncomp      = ncomp_max,
                    validation = validation,
                    scale      = TRUE)

  # Best ncomp by min CV RMSEP
  cv_rmsep  <- pls::RMSEP(model, estimate = "CV")
  best_ncomp <- which.min(cv_rmsep$val[1, 1, ]) - 1L   # -1: intercept-only is index 1

  message(sprintf("[fit_pcr] Best ncomp (min CV RMSEP) = %d", best_ncomp))
  attr(model, "best_ncomp") <- best_ncomp
  model
}

#' Evaluate PCR on test set
#'
#' @param model    Fitted pcr object.
#' @param test     Testing tibble.
#' @param ncomp    Number of components to use. If NULL, uses attr best_ncomp.
#' @return List: $pred, $mse, $ncomp_used.
eval_pcr <- function(model, test, ncomp = NULL) {
  if (is.null(ncomp)) ncomp <- attr(model, "best_ncomp")

  pred <- as.numeric(predict(model, newdata = test, ncomp = ncomp))
  mse  <- mean((pred - test$Price_10K)^2)
  message(sprintf("[eval_pcr] Test MSE (ncomp=%d) = %.4f", ncomp, mse))
  list(pred = pred, mse = mse, ncomp_used = ncomp)
}

# ---------------------------------------------------------------------------
# PLS
# ---------------------------------------------------------------------------

#' Fit PLS regression with cross-validation
#'
#' @param train   Training tibble.
#' @param cfg     Config list.
#' @param formula Model formula. Defaults to polynomial response ~ all numerics.
#' @return Fitted mvr (PLS) object.
fit_pls <- function(train, cfg, formula = NULL) {
  ncomp_max  <- cfg$models$pls$ncomp_max     # 31
  validation <- cfg$models$pls$validation    # "CV"

  if (is.null(formula)) {
    num_preds <- train %>%
      dplyr::select(where(is.numeric), -Price_10K, -dplyr::starts_with("class_")) %>%
      colnames()
    formula <- as.formula(paste("Price_10K ~", paste(num_preds, collapse = " + ")))
  }

  message(sprintf("[fit_pls] Fitting PLS (ncomp=%d, validation=%s) ...",
                  ncomp_max, validation))
  model <- pls::plsr(formula,
                     data       = train,
                     ncomp      = ncomp_max,
                     validation = validation,
                     scale      = TRUE)

  cv_rmsep   <- pls::RMSEP(model, estimate = "CV")
  best_ncomp <- which.min(cv_rmsep$val[1, 1, ]) - 1L

  message(sprintf("[fit_pls] Best ncomp (min CV RMSEP) = %d", best_ncomp))
  attr(model, "best_ncomp") <- best_ncomp
  model
}

#' Evaluate PLS on test set
#'
#' @param model    Fitted plsr object.
#' @param test     Testing tibble.
#' @param ncomp    Number of components. If NULL uses attr best_ncomp.
#' @return List: $pred, $mse, $ncomp_used.
eval_pls <- function(model, test, ncomp = NULL) {
  if (is.null(ncomp)) ncomp <- attr(model, "best_ncomp")

  pred <- as.numeric(predict(model, newdata = test, ncomp = ncomp))
  mse  <- mean((pred - test$Price_10K)^2)
  message(sprintf("[eval_pls] Test MSE (ncomp=%d) = %.4f", ncomp, mse))
  list(pred = pred, mse = mse, ncomp_used = ncomp)
}

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

#' Save RMSEP vs. number of components plot for PCR or PLS
#'
#' @param model Fitted pcr or plsr object.
#' @param cfg   Config list.
#' @param tag   File tag, e.g. "pcr_rmsep" or "pls_rmsep".
save_rmsep_plot <- function(model, cfg, tag = "pcr_rmsep") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  png(out, width = 900, height = 600, res = 120)
  pls::validationplot(model, val.type = "RMSEP",
                      main = paste(toupper(gsub("_.*", "", tag)), "— RMSEP vs. Components"))
  dev.off()
  message(sprintf("[save_rmsep_plot] Saved → %s", out))
}

#' Save variance explained plot
#'
#' @param model Fitted pcr or plsr object.
#' @param cfg   Config list.
#' @param tag   File tag.
save_variance_plot <- function(model, cfg, tag = "pcr_variance") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  png(out, width = 900, height = 600, res = 120)
  pls::explvar(model) |>
    cumsum() |>
    plot(type = "b", xlab = "Components", ylab = "Cumulative Variance Explained (%)",
         main = paste(toupper(gsub("_.*", "", tag)), "— Variance Explained"))
  dev.off()
  message(sprintf("[save_variance_plot] Saved → %s", out))
}
