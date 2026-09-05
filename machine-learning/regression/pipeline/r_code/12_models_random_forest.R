# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

#' Save the OOB error vs. number of trees plot (ranger-compatible)
#'
#' @param model ranger object.
#' @param cfg   Config list.
#' @param tag   File tag.
save_rf_error_plot <- function(model, cfg, tag = "rf_oob_error") {
  
  # Compute OOB MSE manually for regression
  oob_pred   <- model$predictions
  oob_actual <- model$training.data$Price_10K
  oob_mse    <- (oob_actual - oob_pred)^2
  
  df <- data.frame(
    Trees = seq_along(oob_mse),
    OOB_MSE = oob_mse
  )
  
  p <- ggplot(df, aes(x = Trees, y = OOB_MSE)) +
    geom_line(color = "steelblue", linewidth = 1) +
    labs(
      title = paste("Random Forest OOB Error —", tag),
      x = "Trees",
      y = "OOB MSE"
    ) +
    theme_minimal(base_size = 14)
  
  out_path <- file.path(
    here::here(cfg$paths$plot_dir),
    paste0(tag, ".png")
  )
  
  ggsave(out_path, p, width = 8, height = 6)
  message("[save_rf_error_plot] Saved → ", out_path)
}


#' Save the variable importance plot (ranger-compatible)
#'
#' @param model ranger object.
#' @param cfg   Config list.
#' @param tag   File tag.
save_importance_plot <- function(model, cfg, tag = "rf_importance") {
  
  # RANGER MODEL --------------------------------------------------------------
  if ("ranger" %in% class(model)) {
    
    imp <- data.frame(
      Variable   = names(model$variable.importance),
      Importance = model$variable.importance
    )
    
    p <- ggplot(imp, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(
        title = paste("Variable Importance —", tag),
        x = "Predictor",
        y = "Importance"
      ) +
      theme_minimal(base_size = 14)
    
    # RANDOMFOREST MODEL --------------------------------------------------------
  } else if ("randomForest" %in% class(model)) {
    
    imp <- data.frame(
      Variable   = rownames(model$importance),
      Importance = model$importance[, 1]   # IncNodePurity
    )
    
    p <- ggplot(imp, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "darkgreen") +
      coord_flip() +
      labs(
        title = paste("Variable Importance —", tag),
        x = "Predictor",
        y = "IncNodePurity"
      ) +
      theme_minimal(base_size = 14)
    
  } else {
    stop("Unsupported RF model class.")
  }
  
  out_path <- file.path(
    here::here(cfg$paths$plot_dir),
    paste0(tag, ".png")
  )
  
  ggsave(out_path, p, width = 8, height = 6)
  message("[save_importance_plot] Saved → ", out_path)
}
