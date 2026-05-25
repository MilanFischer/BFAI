################################################################################
# worker.R
################################################################################

args <- commandArgs(trailingOnly = TRUE)

run_i <- as.integer(args[1])

runs <- readRDS("runs.rds")

source("pipeline_function.R")

run_pipeline(
  run_i = run_i,
  runs = runs
)