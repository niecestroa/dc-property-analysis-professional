# =============================================================================
# 13_models_svm.R
# Support Vector Machines for binary classification of Price_10K.
# Matches paper's Section: SVM — median split (Under / Over at 44.365).
# Tests four kernels: linear, radial, polynomial, sigmoid.
# Best reported kernel: radial (highest test MSE = 0.6385176 in paper).
# =============================================================================

suppressPackageStartupMessages({
  library(e1071)
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# Helper: numeric predictor matrix for SVM
# ---------------------------------------------------------------------------
.svm_Xy <- function(df, class_col = "class_median") {
  exclude <- c("Price_10K", "class_mean", "class_median")
  X <- df %>%
    dplyr::select(-dplyr::any_of(exclude)) %>%
    dplyr::select(where(is.numeric))
  y <- df[[class_col]]
  list(X = X, y = y)
}

# ---------------------------------------------------------------------------
# Single-kernel SVM
# ---------------------------------------------------------------------------

#' Fit an SVM model with a specified kernel
#'
#' @param train     Training tibble (must have class labels from add_classification_labels()).
#' @param kernel    One of "linear", "radial", "polynomial", "sigmoid".
#' @param class_col Classification target column. Default "class_median" (Under/Over).
#' @param cfg       Config list.
#' @return Fitted e1071::svm object.
fit_svm <- function(train, kernel = "radial", class_col = "class_median", cfg) {
  cost  <- cfg$models$svm$cost_default
  gamma <- cfg$models$svm$gamma_default

  xy <- .svm_Xy(train, class_col)

  message(sprintf("[fit_svm] Fitting SVM | kernel=%s | cost=%.2f | gamma=%.3f",
                  kernel, cost, gamma))

  model <- e1071::svm(
    x      = xy$X,
    y      = xy$y,
    kernel = kernel,
    cost   = cost,
    gamma  = gamma,
    type   = "C-classification",
    scale  = TRUE
  )

  message(sprintf("[fit_svm] Number of support vectors: %d", sum(model$nSV)))
  model
}

#' Evaluate SVM on test set
#'
#' @param model     Fitted svm object.
#' @param test      Testing tibble.
#' @param class_col Classification column.
#' @return List: $pred, $mse (misclassification rate), $confusion.
eval_svm <- function(model, test, class_col = "class_median") {
  xy   <- .svm_Xy(test, class_col)
  pred <- predict(model, newdata = xy$X)
  mse  <- mean(pred != xy$y)
  conf <- table(Predicted = pred, Actual = xy$y)
  message(sprintf("[eval_svm] Misclassification rate = %.7f", mse))
  list(pred = pred, mse = mse, confusion = conf)
}

# ---------------------------------------------------------------------------
# Multi-kernel sweep
# ---------------------------------------------------------------------------

#' Run all four kernels and return a comparison table
#'
#' @param train     Training tibble.
#' @param test      Testing tibble.
#' @param class_col Classification column.
#' @param cfg       Config list.
#' @return Tibble: Kernel, Misclassification_Rate, Correct_Rate.
compare_svm_kernels <- function(train, test, class_col = "class_median", cfg) {
  kernels <- cfg$models$svm$kernels   # ["linear", "radial", "polynomial", "sigmoid"]

  results <- purrr::map_dfr(kernels, function(k) {
    m <- fit_svm(train, kernel = k, class_col = class_col, cfg = cfg)
    e <- eval_svm(m, test, class_col = class_col)
    tibble::tibble(
      Kernel                  = k,
      Misclassification_Rate  = e$mse,
      Correct_Rate            = 1 - e$mse
    )
  })

  message("\n[compare_svm_kernels] Kernel comparison:")
  print(results)
  invisible(results)
}

# ---------------------------------------------------------------------------
# 2-D scatter plot (Latitude × Longitude coloured by class)
# Matches paper's visualisation of Under / Over classifications
# ---------------------------------------------------------------------------

#' Save 2-D scatter plot of Latitude vs Longitude coloured by class
#'
#' @param df        Tibble with Latitude, Longitude, and class_median columns.
#' @param class_col Column to colour by. Default "class_median".
#' @param cfg       Config list.
#' @param tag       File tag.
save_svm_scatter <- function(df, class_col = "class_median", cfg, tag = "svm_scatter") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)

  p <- ggplot2::ggplot(
    df %>% dplyr::sample_n(min(5000, nrow(df))),  # subsample for speed
    ggplot2::aes(x = Longitude, y = Latitude, colour = .data[[class_col]])
  ) +
    ggplot2::geom_point(alpha = 0.4, size = 0.8) +
    ggplot2::scale_colour_manual(values = c("steelblue", "firebrick")) +
    ggplot2::labs(
      title  = "DC Properties — Price Classification (Lat × Lon)",
      colour = class_col
    ) +
    ggplot2::theme_bw()

  ggplot2::ggsave(out, p, width = 8, height = 6, dpi = 150)
  message(sprintf("[save_svm_scatter] Saved → %s", out))
}
