################################################################################
# run_all.R
################################################################################

library(tidyverse)
library(callr)

Sys.setenv(CLI_NUM_COLORS = 1)
options(
  crayon.enabled = FALSE,
  cli.num_colors = 1,
  cli.dynamic = FALSE
)

start_ID <- 13

runs <- tidyr::crossing(
  cor_thresh = seq(0.7, 0.95, 0.05),
  use_country = c(TRUE),
  use_meteo = c(TRUE),
  use_winter = c(TRUE),
  use_year = c(FALSE),
  metamodel = c(FALSE),
  use_perturbation_audit = c(TRUE),
  use_synthetic_audit = c(FALSE),
  use_scenario_audit = c(TRUE),
  scenario_bfai_multiplier = c(10),
  grid_ini = c(50), # 50
  grid_race = c(100), # 100
  n_ens_reps = c(20), # 20
  num_cores_tune = 14,
  num_cores_plot = 18
) |>
  mutate(
    run_ID = start_ID - 1 + row_number(),
    out_path = file.path(
      "./outputs",
      paste0("out_", sprintf("%03d", run_ID))
    )
  )

################################################################################
# Save runs table so pipeline_worker.R can read it
################################################################################

saveRDS(runs, "runs.rds")

################################################################################
# Launch each run in a fresh R session
################################################################################

for (run_i in 1) {
  
  message(
    "\n=====================================================\n",
    "Starting run ", run_i, " / ", nrow(runs),
    "\n=====================================================\n"
  )
  
  result <- tryCatch({
    
    dir.create(runs$out_path[run_i], recursive = TRUE, showWarnings = FALSE)
    
    log_file <- file.path(
      runs$out_path[run_i],
      "pipeline_log.txt"
    )
    
    callr::rscript(
      script = "pipeline_worker.R",
      cmdargs = as.character(run_i),
      stdout = log_file,
      stderr = log_file
    )
    
    TRUE
    
  }, error = function(e) {
    
    message(
      "\nERROR in run ", run_i, ":\n",
      e$message
    )
    
    FALSE
  })
  
  if (result) {
    
    message(
      "\nFinished run ", run_i, "\n"
    )
    
  } else {
    
    message(
      "\nRun ", run_i, " FAILED\n"
    )
    
  }
  
}

################################################################################
# Cleanup
################################################################################

if (file.exists("runs.rds")) {
  file.remove("runs.rds")
}

message("\nALL RUNS FINISHED\n")