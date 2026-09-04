# =============================================================================
# .Rprofile — DC Property ML Project
# Runs automatically when RStudio opens this project (.Rproj).
# =============================================================================

# Use {here} to anchor all paths to the project root
if (requireNamespace("here", quietly = TRUE)) {
  here::i_am("dc-property-ml.Rproj")
}

# Suppress startup messages for cleaner console
suppressPackageStartupMessages({
  if (requireNamespace("dplyr",   quietly = TRUE)) library(dplyr)
  if (requireNamespace("ggplot2", quietly = TRUE)) library(ggplot2)
})

message("DC Property ML project loaded. Run `source('run_all.R')` to execute the full pipeline.")
message("Or knit notebooks/01_eda.Rmd / notebooks/02_modeling.Rmd for interactive analysis.")
