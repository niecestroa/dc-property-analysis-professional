# =============================================================================
# 12_models_random_forest.R
# Random Forest + Bagging for regression on DC Property Price_10K.
# Matches paper findings: optimal RF at 143 trees, mtry=1, 54.81% var explained.
# Most important predictor: Bathrooms (optimal); Longitude (full forest).
# =============================================================================

suppressPackageStartupMessages({
  library(randomForest)
  library(ranger)
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# Helper: build predictor formula
# ---------------------------------------------------------------------------
.rf_formula <- function(train, extra_exclude = character(0)) {
  exclude <- c("Price_10K", "class_mean", "class_median", extra_exclude)
  preds   <- setdiff(colnames(train), exclude)
  # Drop non-numeric & factor cols with too many levels (RF handles factors natively)
  as.formula(paste("Price_10K ~", paste(preds, collapse = " + ")))
}

# ---------------------------------------------------------------------------
# Full Random Forest
# ---------------------------------------------------------------------------

#' Fit a full random forest (all predictors, ntree from cfg)
#'
#' @param train  Training tibble.
#' @param cfg    Config list.
#' @param ntree  Number of trees. Sourced from cfg if NULL.
#' @return randomForest object.
fit_random_forest <- function(train, cfg, ntree = NULL) {
  
  library(ranger)
  
  if (is.null(ntree)) {
    ntree <- max(cfg$models$random_forest$ntree_values)
  }
  
  if (ntree > 200) ntree <- 200
  
  exclude <- c("class_mean", "class_median")
  train_rf <- train %>% dplyr::select(-dplyr::any_of(exclude))
  
  p <- ncol(train_rf) - 1
  mtry_val <- floor(sqrt(p))
  
  message(sprintf("[fit_random_forest] Fitting RANGER RF (ntree=%d, mtry=%d) ...",
                  ntree, mtry_val))
  
  set.seed(cfg$project$seed)
  
  model <- ranger(
    formula    = Price_10K ~ .,
    data       = train_rf,
    num.trees  = ntree,
    mtry       = mtry_val,
    importance = "impurity",
    oob.error  = TRUE
  )
  
  pct_var <- round(100 * (1 - model$prediction.error), 2)
  message(sprintf("[fit_random_forest] %% Var explained (OOB) = %.2f%%", pct_var))
  
  model
}


# ---------------------------------------------------------------------------
# Optimal Random Forest (paper: 143 trees, mtry=1, predictors = numeric only)
# ---------------------------------------------------------------------------

#' Fit the optimal random forest found in the paper
#'
#' Predictors: Bathrooms, Rooms, Bedrooms, Stories, Kitchens, Fireplaces
#' (paper's subset used for the "optimal" CV search).
#'
#' @param train  Training tibble.
#' @param cfg    Config list.
#' @return randomForest object.
fit_optimal_rf <- function(train, cfg) {
  optimal_preds <- c("Bathrooms", "Rooms", "Bedrooms", "Stories", "Kitchens", "Fireplaces")
  optimal_preds <- intersect(optimal_preds, colnames(train))

  ntree <- cfg$models$random_forest$ntree_values[1]   # 143
  mtry  <- cfg$models$random_forest$mtry_range[1]     # 1

  message(sprintf("[fit_optimal_rf] ntree=%d | mtry=%d | preds: %s",
                  ntree, mtry, paste(optimal_preds, collapse = ", ")))

  set.seed(cfg$project$seed)
  model <- randomForest::randomForest(
    x          = train[, optimal_preds, drop = FALSE],
    y          = train$Price_10K,
    ntree      = ntree,
    mtry       = mtry,
    importance = TRUE
  )

  pct_var <- round(100 * mean(model$rsq), 2)
  message(sprintf("[fit_optimal_rf] %% Var explained (OOB) = %.2f%%", pct_var))
  model
}

# ---------------------------------------------------------------------------
# Bagging
# ---------------------------------------------------------------------------

#' Bagging = Random Forest with mtry = p (all predictors at each split)
#'
#' @param train  Training tibble.
#' @param cfg    Config list.
#' @return randomForest object (bagged).
fit_bagging <- function(train, cfg) {
  exclude   <- c("class_mean", "class_median", "Price_10K")
  n_preds   <- ncol(train) - length(intersect(colnames(train), exclude))
  ntree     <- cfg$models$bagging$ntree   # 500

  message(sprintf("[fit_bagging] Bagging: ntree=%d | mtry=%d (all predictors)", ntree, n_preds))
  set.seed(cfg$project$seed)

  train_bag <- train %>% dplyr::select(-dplyr::any_of(c("class_mean", "class_median")))

  model <- randomForest::randomForest(
    Price_10K ~ .,
    data       = train_bag,
    ntree      = ntree,
    mtry       = n_preds,
    importance = TRUE
  )
  message(sprintf("[fit_bagging] OOB MSE = %.4f", tail(model$mse, 1)))
  model
}

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

#' Compute test MSE for a random forest model
#'
#' @param model   randomForest object.
#' @param test    Testing tibble.
#' @return List: $pred, $mse.
eval_rf <- function(model, test) {
  # Align test columns with training columns
  vars_used <- rownames(model$importance)
  test_x    <- test[, intersect(vars_used, colnames(test)), drop = FALSE]

  pred <- predict(model, newdata = test_x)
  mse  <- mean((pred - test$Price_10K)^2)
  message(sprintf("[eval_rf] Test MSE = %.4f", mse))
  list(pred = pred, mse = mse)
}

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

#' Save the OOB error vs. number of trees plot
#'
#' @param model randomForest object.
#' @param cfg   Config list.
#' @param tag   File tag.
save_rf_error_plot <- function(model, cfg, tag = "rf_oob_error") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)

  err_df <- data.frame(
    Trees = seq_along(model$mse),
    OOB_MSE = model$mse
  )

  p <- ggplot2::ggplot(err_df, ggplot2::aes(x = Trees, y = OOB_MSE)) +
    ggplot2::geom_line(color = "steelblue", linewidth = 0.8) +
    ggplot2::labs(title = "Random Forest — OOB MSE vs. Number of Trees",
                  x = "Number of Trees", y = "OOB MSE") +
    ggplot2::theme_bw()

  ggplot2::ggsave(out, p, width = 8, height = 5, dpi = 150)
  message(sprintf("[save_rf_error_plot] Saved → %s", out))
}

#' Save the variable importance plot
#'
#' @param model randomForest object with importance=TRUE.
#' @param cfg   Config list.
#' @param tag   File tag.
save_importance_plot <- function(model, cfg, tag = "rf_importance") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  png(out, width = 900, height = 600, res = 120)
  randomForest::varImpPlot(model,
                           main = "Random Forest — Variable Importance",
                           type = 2)   # Node purity (IncNodePurity)
  dev.off()
  message(sprintf("[save_importance_plot] Saved → %s", out))
}
