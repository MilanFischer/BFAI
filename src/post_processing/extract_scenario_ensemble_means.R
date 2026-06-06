library(tidyverse)

root <- "./outputs"
outs <- sprintf("out_%03d", 1:12)

extract_all_grey_curve_data <- function(out_id) {
  
  plot_dir <- file.path(root, out_id, "plot_data")
  clean_file <- file.path(root, out_id, "clean_data_for_plots.rds")
  
  if (!dir.exists(plot_dir) || !file.exists(clean_file)) {
    message("Skipping ", out_id, " - missing plot_data or clean_data_for_plots.rds")
    return(NULL)
  }
  
  ensemble_files <- list.files(
    plot_dir,
    pattern = "_ensemble_predictions\\.rds$",
    full.names = TRUE
  )
  
  if (length(ensemble_files) == 0) {
    message("Skipping ", out_id, " - no ensemble prediction files")
    return(NULL)
  }
  
  clean_data <- readRDS(clean_file)
  
  map_dfr(ensemble_files, function(file) {
    
    country_code <- basename(file) |>
      str_remove("_ensemble_predictions\\.rds$")
    
    ens <- readRDS(file)
    
    ssp_values <- ens |>
      filter(!is.na(Scenario)) |>
      distinct(Scenario) |>
      pull(Scenario)
    
    map_dfr(ssp_values, function(ssp_use) {
      
      baseline <- clean_data |>
        filter(Country == country_code, Year %in% 1981:2010) |>
        summarise(
          Period = "1981-2010",
          Period_num = 1996,
          predicted_BFA1000 = exp(mean(log_BFA1000, na.rm = TRUE)),
          .groups = "drop"
        )
      
      future_mean <- ens |>
        filter(Scenario == ssp_use) |>
        group_by(Period) |>
        summarise(
          predicted_BFA1000 = mean(predicted_BFA1000, na.rm = TRUE),
          .groups = "drop"
        ) |>
        mutate(
          Period = as.character(Period),
          Period_num = case_when(
            Period == "2030" ~ 2030,
            Period == "2050" ~ 2050,
            Period == "2070" ~ 2070,
            Period == "2085" ~ 2085,
            TRUE ~ NA_real_
          )
        ) |>
        filter(!is.na(Period_num))
      
      bind_rows(baseline, future_mean) |>
        mutate(
          out_id = out_id,
          country = country_code,
          ssp = ssp_use,
          statistic = "mean across GCMs"
        ) |>
        select(out_id, country, ssp, Period, Period_num, predicted_BFA1000, statistic)
    })
  })
}

grey_curve_points_all <- map_dfr(outs, extract_all_grey_curve_data)

write_csv(
  grey_curve_points_all,
  file.path(root, "grey_curve_input_points_all_outputs_all_countries_all_ssps.csv")
)

grey_curve_points_all


################################################################################
extract_all_gcm_points <- function(out_id) {
  
  plot_dir <- file.path(root, out_id, "plot_data")
  
  if (!dir.exists(plot_dir)) {
    message("Skipping ", out_id, " - missing plot_data")
    return(NULL)
  }
  
  ensemble_files <- list.files(
    plot_dir,
    pattern = "_ensemble_predictions\\.rds$",
    full.names = TRUE
  )
  
  map_dfr(ensemble_files, function(file) {
    
    country_code <- basename(file) |>
      str_remove("_ensemble_predictions\\.rds$")
    
    ens <- readRDS(file)
    
    ens |>
      filter(!is.na(Scenario)) |>
      group_by(Scenario, GCM, Period) |>
      summarise(
        predicted_BFA1000 = mean(predicted_BFA1000, na.rm = TRUE),
        .groups = "drop"
      ) |>
      mutate(
        out_id = out_id,
        country = country_code
      ) |>
      select(out_id, country, Scenario, GCM, Period, predicted_BFA1000)
  })
}

gcm_points_all <- map_dfr(outs, extract_all_gcm_points)

write_csv(
  gcm_points_all,
  file.path(root, "gcm_points_all_outputs_all_countries_all_ssps.csv")
)

gcm_points_all