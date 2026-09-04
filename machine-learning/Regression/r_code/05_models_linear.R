# =============================================================================
# 05_models_linear.R
# Simple and multiple linear regression models.
# Includes variable selection (AIC/BIC stepwise, forward, backward) and
# partial F-test helpers, reproducing the paper's Section: Linear Regression.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(MASS)    # stepAIC
})

# ---------------------------------------------------------------------------
# Linear regression formula (final model from paper)
# ---------------------------------------------------------------------------
LINEAR_FORMULA <- Price_10K ~
  Bathrooms + Rooms + Bedrooms + Stories + Qualified +
  Grade + Kitchens + Fireplaces +
  Ward + Latitude + Longitude +
  AYB_age + EYB_age + Remodel_age +
  Condition

#' Fit the final multiple linear regression model
#'
#' @param train  Training tibble from make_split().
#' @param formula Optional formula override (defaults to paper's final model).
#' @return A fitted lm object.
fit_linear <- function(train, formula = LINEAR_FORMULA) {
  message("[fit_linear] Fitting multiple linear regression ...")
  model <- lm(formula, data = train)
  message(sprintf("[fit_linear] Adj. R2 = %.4f | RSE = %.4f",
                  summary(model)$adj.r.squared,
                  summary(model)$sigma))
  model
}

#' Variable selection via stepwise AIC (both directions)
#'
#' @param train   Training tibble.
#' @param formula Starting formula.
#' @return Best lm model by AIC.
select_by_aic <- function(train, formula = LINEAR_FORMULA) {
  message("[select_by_aic] Running stepwise AIC selection ...")
  full  <- lm(formula, data = train)
  best  <- MASS::stepAIC(full, direction = "both", trace = FALSE)
  message(sprintf("[select_by_aic] Best AIC model — terms: %d | AIC: %.1f",
                  length(coef(best)), AIC(best)))
  best
}

#' Variable selection via stepwise BIC
#'
#' @param train   Training tibble.
#' @param formula Starting formula.
#' @return Best lm model by BIC.
select_by_bic <- function(train, formula = LINEAR_FORMULA) {
  message("[select_by_bic] Running stepwise BIC selection ...")
  full  <- lm(formula, data = train)
  n     <- nrow(train)
  best  <- MASS::stepAIC(full, direction = "both", k = log(n), trace = FALSE)
  message(sprintf("[select_by_bic] Best BIC model — terms: %d | BIC: %.1f",
                  length(coef(best)), BIC(best)))
  best
}

#' Run forward selection from an intercept-only model
#'
#' @param train   Training tibble.
#' @param formula Upper scope formula.
#' @return Best lm from forward selection.
select_forward <- function(train, formula = LINEAR_FORMULA) {
  message("[select_forward] Running forward selection ...")
  null <- lm(Price_10K ~ 1, data = train)
  best <- MASS::stepAIC(null,
                        scope = list(upper = formula, lower = ~1),
                        direction = "forward", trace = FALSE)
  best
}

#' Run backward selection from the full model
#'
#' @param train   Training tibble.
#' @param formula Full model formula.
#' @return Best lm from backward selection.
select_backward <- function(train, formula = LINEAR_FORMULA) {
  message("[select_backward] Running backward selection ...")
  full <- lm(formula, data = train)
  best <- MASS::stepAIC(full, direction = "backward", trace = FALSE)
  best
}

#' Partial F-test between nested models
#'
#' Replicates the paper's H0: β_Condition = 0 etc.
#'
#' @param reduced  lm object: reduced model.
#' @param full     lm object: full model.
#' @return anova output (printed) and invisible anova table.
partial_f_test <- function(reduced, full) {
  message("[partial_f_test] Running partial F-test ...")
  result <- anova(reduced, full)
  print(result)
  invisible(result)
}

#' Variance Inflation Factor check
#'
#' @param model Fitted lm.
#' @return Named numeric vector of VIFs.
check_vif <- function(model) {
  if (!requireNamespace("car", quietly = TRUE)) {
    stop("Install the 'car' package: install.packages('car')")
  }
  vif_vals <- car::vif(model)
  high_vif  <- vif_vals[vif_vals > 5]
  if (length(high_vif) > 0) {
    warning(sprintf("[check_vif] %d predictors have VIF > 5:\n%s",
                    length(high_vif),
                    paste(names(high_vif), round(high_vif, 2), sep = " = ", collapse = "\n  ")))
  } else {
    message("[check_vif] All VIFs < 5 — no multicollinearity concern.")
  }
  vif_vals
}
