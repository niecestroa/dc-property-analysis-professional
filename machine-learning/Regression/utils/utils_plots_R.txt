# =============================================================================
# utils/utils_plots.R
# Reusable plotting helpers for EDA and model diagnostics.
# All plots saved to cfg$paths$plot_dir automatically.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

#' Histogram of Price_10K with mean and median lines
#'
#' @param df  Cleaned tibble.
#' @param cfg Config list.
save_price_histogram <- function(df, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir), "eda_price_histogram.png")
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)

  mn <- mean(df$Price_10K, na.rm = TRUE)
  md <- median(df$Price_10K, na.rm = TRUE)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = Price_10K)) +
    ggplot2::geom_histogram(bins = 80, fill = "steelblue", colour = "white", alpha = 0.85) +
    ggplot2::geom_vline(xintercept = mn, colour = "firebrick",
                        linetype = "dashed", linewidth = 1) +
    ggplot2::geom_vline(xintercept = md, colour = "darkgreen",
                        linetype = "dashed", linewidth = 1) +
    ggplot2::annotate("text", x = mn + 5, y = Inf, vjust = 2,
                      label = sprintf("Mean=%.3f", mn), colour = "firebrick", size = 3.5) +
    ggplot2::annotate("text", x = md - 5, y = Inf, vjust = 4,
                      label = sprintf("Median=%.3f", md), colour = "darkgreen", size = 3.5,
                      hjust = 1) +
    ggplot2::labs(title = "DC Property — Price Distribution (in $10K)",
                  x = "Price ($10K)", y = "Count") +
    ggplot2::theme_bw()

  ggplot2::ggsave(out, p, width = 9, height = 5, dpi = 150)
  message(sprintf("[save_price_histogram] Saved → %s", out))
  invisible(p)
}

#' Boxplots of Price_10K by a categorical variable
#'
#' @param df      Tibble.
#' @param cat_var Categorical column name (e.g. "Grade", "Ward", "Condition").
#' @param cfg     Config list.
save_price_by_category <- function(df, cat_var, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir),
                   paste0("eda_price_by_", tolower(cat_var), ".png"))

  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data[[cat_var]], y = Price_10K,
                                         fill = .data[[cat_var]])) +
    ggplot2::geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3, show.legend = FALSE) +
    ggplot2::labs(title = paste("Price_10K by", cat_var),
                  x = cat_var, y = "Price ($10K)") +
    ggplot2::theme_bw() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))

  ggplot2::ggsave(out, p, width = 9, height = 5, dpi = 150)
  message(sprintf("[save_price_by_category] Saved → %s", out))
  invisible(p)
}

#' Scatter matrix of key numeric predictors vs Price_10K
#'
#' @param df  Tibble.
#' @param cfg Config list.
save_scatter_matrix <- function(df, cfg) {
  if (!requireNamespace("GGally", quietly = TRUE)) {
    warning("[save_scatter_matrix] Install GGally: install.packages('GGally')")
    return(invisible(NULL))
  }
  out <- file.path(here::here(cfg$paths$plot_dir), "eda_scatter_matrix.png")
  vars <- c("Price_10K", "Bathrooms", "Rooms", "Bedrooms",
            "Fireplaces", "AYB_age", "EYB_age")
  vars <- intersect(vars, colnames(df))

  p <- GGally::ggpairs(df[, vars], progress = FALSE,
                        upper = list(continuous = GGally::wrap("cor", size = 3)),
                        lower = list(continuous = GGally::wrap("points", alpha = 0.1, size = 0.3)),
                        title = "Scatter Matrix — Key Predictors vs Price_10K")

  ggplot2::ggsave(out, p, width = 12, height = 12, dpi = 120)
  message(sprintf("[save_scatter_matrix] Saved → %s", out))
  invisible(p)
}

#' Geographic scatter: Lat × Lon coloured by Price_10K
#'
#' @param df  Tibble with Latitude and Longitude columns.
#' @param cfg Config list.
save_geo_price_map <- function(df, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir), "eda_geo_price_map.png")

  p <- ggplot2::ggplot(
    df %>% dplyr::sample_n(min(10000, nrow(df))),
    ggplot2::aes(x = Longitude, y = Latitude, colour = Price_10K)
  ) +
    ggplot2::geom_point(alpha = 0.5, size = 0.6) +
    ggplot2::scale_colour_viridis_c(option = "plasma", name = "Price ($10K)") +
    ggplot2::labs(title = "DC Property Prices — Geographic Distribution") +
    ggplot2::theme_bw()

  ggplot2::ggsave(out, p, width = 8, height = 7, dpi = 150)
  message(sprintf("[save_geo_price_map] Saved → %s", out))
  invisible(p)
}

#' Correlation heatmap for numeric predictors
#'
#' @param df  Tibble.
#' @param cfg Config list.
save_correlation_heatmap <- function(df, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir), "eda_correlation_heatmap.png")

  num_df  <- df %>% dplyr::select(where(is.numeric), -dplyr::starts_with("class_"))
  cor_mat <- cor(num_df, use = "pairwise.complete.obs")
  cor_long <- as.data.frame(as.table(cor_mat)) %>%
    dplyr::rename(Var1 = Var1, Var2 = Var2, Correlation = Freq)

  p <- ggplot2::ggplot(cor_long, ggplot2::aes(x = Var1, y = Var2, fill = Correlation)) +
    ggplot2::geom_tile(colour = "white") +
    ggplot2::scale_fill_gradient2(low = "firebrick", high = "steelblue",
                                   mid = "white", midpoint = 0, limits = c(-1, 1)) +
    ggplot2::labs(title = "Correlation Heatmap — Numeric Predictors") +
    ggplot2::theme_bw() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
                   axis.title  = ggplot2::element_blank())

  ggplot2::ggsave(out, p, width = 10, height = 9, dpi = 150)
  message(sprintf("[save_correlation_heatmap] Saved → %s", out))
  invisible(p)
}

# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

#' Save residual plots for an lm model (Residuals vs Fitted + Q-Q)
#'
#' @param model Fitted lm object.
#' @param cfg   Config list.
#' @param tag   File tag.
save_residual_plots <- function(model, cfg, tag = "linear_diagnostics") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  png(out, width = 1200, height = 600, res = 120)
  par(mfrow = c(1, 2))
  plot(model, which = c(1, 2), ask = FALSE)
  dev.off()
  message(sprintf("[save_residual_plots] Saved → %s", out))
}
