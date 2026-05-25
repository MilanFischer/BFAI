library(tidyverse)
library(furrr)
library(future)

source("./src/log_message.R")
source("./src/colors.R")

args <- commandArgs(trailingOnly = TRUE)
out_path <- args[1]

clean_data <- readRDS(file.path(out_path, "clean_data_for_plots.rds"))
calibration_data <- readRDS(file.path(out_path, "calibration_data_for_plots.rds"))
scenario_settings <- readRDS(file.path(out_path, "scenario_settings.rds"))
use_country <- readRDS(file.path(out_path, "use_country.rds"))
num_cores_plot <- readRDS(file.path(out_path, "num_cores_plot.rds"))

dir.create(file.path(out_path, "scenarios"), recursive = TRUE, showWarnings = FALSE)

#-----------------------
# Scenario plot function
make_country_scenario_plots <- function(country_code, ssp_use) {
  
  bfa_reference <- clean_data |>
    filter(
      Country == country_code,
      Year %in% 1991:2024
    ) |>
    summarise(
      BFA1000_1991_2024 = exp(mean(log_BFA1000, na.rm = TRUE))
    ) |>
    pull(BFA1000_1991_2024)
  
  final_prediction_annual <- readRDS(
    file.path(out_path, "plot_data", paste0(country_code, "_scenario_predictions.rds"))
  )
  
  ens_scen_annual <- readRDS(
    file.path(out_path, "plot_data", paste0(country_code, "_ensemble_predictions.rds"))
  )
  
  final_prediction <- final_prediction_annual |>
    filter(Scenario == ssp_use) |>
    group_by(Scenario, GCM, Period, ML_model) |>
    summarise(
      predicted_BFA1000 = mean(predicted_BFA1000, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      Period = as.character(Period),
      BFA1000_1991_2024 = bfa_reference
    ) |>
    rename(
      Climate_model = GCM
    ) |>
    mutate(
      Climate_model = factor(
        Climate_model,
        levels = c("Baseline", "cmcc-esm2", "ec-earth3", "gfdl-esm4",
                   "mpi-esm1-2-hr", "mri-esm2-0", "taiesm1")
      ),
      Period_num = case_when(
        Period == "2030" ~ 2030,
        Period == "2050" ~ 2050,
        Period == "2070" ~ 2070,
        Period == "2085" ~ 2085,
        TRUE ~ NA_real_
      )
    )
  
  baseline_prediction <- calibration_data |>
    filter(Country == country_code) |>
    mutate(
      Period = case_when(
        # Year %in% 1961:1990 ~ "1961-1990",
        Year %in% 1981:2010 ~ "1981-2010",
        # Year %in% 1991:2020 ~ "1991-2020",
        TRUE ~ NA_character_
      )
    ) |>
    filter(!is.na(Period)) |>
    group_by(Period) |>
    summarise(
      predicted_BFA1000 = exp(mean(log_BFA1000, na.rm = TRUE)),
      BFA1000_1991_2024 = bfa_reference,
      .groups = "drop"
    ) |>
    crossing(
      ML_model = final_prediction |>
        distinct(ML_model) |>
        pull(ML_model)
    ) |>
    mutate(
      Climate_model = "Baseline"
    ) |>
    select(predicted_BFA1000, Period, Climate_model, ML_model, BFA1000_1991_2024)
  
  baseline_prediction |>
    count(ML_model, Period)
  
  
  final_prediction <- final_prediction |>
    select(predicted_BFA1000, Period, Climate_model, ML_model, BFA1000_1991_2024) |>
    bind_rows(baseline_prediction) |>
    mutate(
      Climate_model = factor(
        Climate_model,
        levels = c("Baseline", "cmcc-esm2", "ec-earth3", "gfdl-esm4",
                   "mpi-esm1-2-hr", "mri-esm2-0", "taiesm1")
      ),
      Period_num = case_when(
        # Period == "1961-1990" ~ 1976,
        Period == "1981-2010" ~ 1996,
        # Period == "1991-2020" ~ 2006,
        Period == "2030"      ~ 2030,
        Period == "2050"      ~ 2050,
        Period == "2070"      ~ 2070,
        Period == "2085"      ~ 2085,
        TRUE ~ NA_real_
      )
    )
  
  # Compute per-ML-model smoothed lines
  trend_data <- final_prediction |>
    mutate(period_type = if_else(Period %in% c("1981-2010"), "baseline", "future")) |>
    group_by(ML_model, period_type, Period_num) |>
    summarise(predicted_BFA1000 = if_else(
      period_type == "future",
      mean(predicted_BFA1000, na.rm = TRUE),
      predicted_BFA1000[1]  # use actual point from baseline
    ), .groups = "drop") |>
    group_by(ML_model) |>
    arrange(Period_num) |>
    group_modify(~{
      
      tmp <- .x |>
        filter(is.finite(predicted_BFA1000), is.finite(Period_num))
      
      if (nrow(tmp) < 4 || n_distinct(tmp$Period_num) < 4) {
        return(tibble())
      }
      
      smoothed <- tibble(
        Period_num = seq(min(tmp$Period_num), max(tmp$Period_num), length.out = 100)
      )
      
      fit <- loess(predicted_BFA1000 ~ Period_num, data = tmp, span = 0.8)
      
      smoothed |>
        mutate(
          predicted_BFA1000 = predict(fit, newdata = smoothed),
          ML_model = unique(tmp$ML_model)
        ) |>
        filter(is.finite(predicted_BFA1000))
    }) |>
    ungroup()
  
  # x_labels <- c(1970, 1990, 2010, 2030, 2050, 2070, 2090)
  x_labels <- c(1990, 2010, 2030, 2050, 2070, 2090)
  
  scenario_p <- ggplot(final_prediction, aes(x = Period_num, y = predicted_BFA1000, color = Climate_model)) +
    
    # Add ML-model specific smoothed lines
    geom_line(
      data = trend_data,
      aes(x = Period_num, y = predicted_BFA1000, group = ML_model, linetype = "Ensemble trend"),
      inherit.aes = FALSE,
      color = "#2b2b2b", size = 1.2, alpha = 0.3
    ) +
    
    # Reference line
    geom_hline(
      aes(yintercept = BFA1000_1991_2024, linetype = "Observed (1991–2024)"),
      data = final_prediction |> filter(Period == "1981-2010"),
      color = "#2b2b2b", size = 0.8
    ) +
    
    # ML model predictions
    geom_point(size = 3, alpha = 0.8) +
    
    # Facet by ML model
    facet_wrap(~ ML_model) +
    
    # X-axis as real timeline
    scale_x_continuous(
      breaks = x_labels,
      labels = x_labels,
      limits = c(1988, 2092)
    ) +
    
    # Color legend for Climate Models
    scale_color_manual(
      values = custom_colors_GCM,
      breaks = c("Baseline", "cmcc-esm2", "ec-earth3", "gfdl-esm4",
                 "mpi-esm1-2-hr", "mri-esm2-0", "taiesm1"),
      name = "Climate data",
      drop = FALSE,
      guide = guide_legend(order = 1)
    ) +
    
    # Linetype legend
    scale_linetype_manual(
      values = c("Observed (1991–2024)" = "dashed", "Ensemble trend" = "solid"),
      # values = c("Ensemble trend" = "solid"),
      name = NULL,
      guide = guide_legend(
        override.aes = list(
          color = "#2b2b2b",
          size = c(0.8, 1.2),
          alpha = c(0.3, 1)
          # size = c(0.8),
          # alpha = c(0.3)
        ),
        order = 2,
        keywidth = 1.8
      )
    ) +
    
    # Labels and theme
    labs(
      title = "Predicted BFAI Across Climate Models and Machine Learning Models",
      x = "Year",
      y = "Predicted BFAI"
    ) +
    theme_bw() +
    theme(
      legend.position = "right",
      legend.text = element_text(size = 9),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.title.x = element_text(margin = margin(t = 15)),
      strip.text = element_text(face = "bold")
    ) #+
  #coord_cartesian(ylim = c(0.05, 0.35))
  
  plot_name <- if (use_country == TRUE) {
    paste0(out_path, "/scenarios/scenarios_", country_code, "_", ssp_use, "_all_models_predictions_for_scenarios_country.png")
  } else {
    paste0(out_path, "/scenarios/scenarios_", country_code, "_", ssp_use, "_all_models_predictions_for_scenarios.png")
  }
  ggsave(plot_name, plot = scenario_p, width = 3 * 140, height = 2 * 120, dpi = 600, units = 'mm', device = ragg::agg_png)
  
  #-------------------------------------------------------------------------------
  
  ens_scen <- ens_scen_annual |>
    filter(Scenario == ssp_use) |>
    group_by(GCM, Period, ML_model) |>
    summarise(
      predicted_BFA1000 = mean(predicted_BFA1000, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      Period = as.character(Period),
      BFA1000_1991_2024 = bfa_reference
    ) |>
    rename(
      Climate_model = GCM
    )
  
  baseline_ens <- clean_data |>
    filter(Country == country_code) |>
    mutate(
      Period = case_when(
        # Year %in% 1961:1990 ~ "1961-1990",
        Year %in% 1981:2010 ~ "1981-2010",
        # Year %in% 1990:2024 ~ "1990-2024",
        TRUE ~ NA_character_
      )
    ) |>
    filter(!is.na(Period)) |>
    group_by(Period) |>
    summarise(
      predicted_BFA1000 = exp(mean(log_BFA1000, na.rm = TRUE)),
      BFA1000_1991_2024 = bfa_reference,
      .groups = "drop"
    ) |>
    mutate(
      Climate_model = "Baseline",
      ML_model = "ensemble mean"
    )
  
  ens_scen <- ens_scen |>
    bind_rows(baseline_ens) |>
    mutate(
      Climate_model = factor(
        Climate_model,
        levels = c("Baseline", "cmcc-esm2", "ec-earth3", "gfdl-esm4",
                   "mpi-esm1-2-hr", "mri-esm2-0", "taiesm1")
      ),
      Period_num = case_when(
        # Period == "1961-1990" ~ 1976,
        Period == "1981-2010" ~ 1996,
        # Period == "1990-2024" ~ 2007,
        Period == "2030"      ~ 2030,
        Period == "2050"      ~ 2050,
        Period == "2070"      ~ 2070,
        Period == "2085"      ~ 2085,
        TRUE ~ NA_real_
      )
    )
  
  mean_scenarios <- ens_scen |>
    filter(Period %in% c("2030", "2050", "2070", "2085")) |>
    group_by(Period, Period_num) |>
    summarise(
      predicted_BFA1000 = mean(predicted_BFA1000, na.rm = TRUE),
      .groups = "drop"
    )
  
  baseline_means <- ens_scen |>
    # filter(Period %in% c("1961-1990", "1981-2010", "1991-2020")) |>
    filter(Period %in% c("1981-2010")) |>
    select(Period, Period_num, predicted_BFA1000) |>
    distinct()
  
  line_data <- bind_rows(baseline_means, mean_scenarios)
  
  # Define numeric positions for each period
  period_map <- c(
    # "1961-1990" = 1976,
    "1981-2010" = 1996,
    # "1991-2024" = 2007,
    "2030"      = 2030,
    "2050"      = 2050,
    "2070"      = 2070,
    "2085"      = 2085
  )
  
  # Fit loess model for smoothed line
  line_data <- line_data |>
    dplyr::filter(
      is.finite(predicted_BFA1000),
      is.finite(Period_num)
    )
  loess_fit <- loess(predicted_BFA1000 ~ Period_num, data = line_data)
  smoothed_data <- data.frame(
    Period_num = seq(min(line_data$Period_num), max(line_data$Period_num), length.out = 100)
  )
  smoothed_data$predicted_BFA1000 <- predict(loess_fit, newdata = smoothed_data)
  
  # Define desired tick marks and labels for x-axis
  x_labels <- c(1990, 2010, 2030, 2050, 2070, 2090)
  
  # Create plot
  scenario_p <- ggplot(ens_scen, aes(x = Period_num, y = predicted_BFA1000, color = Climate_model)) +
    
    # Ensemble trend line
    geom_line(
      data = smoothed_data,
      aes(x = Period_num, y = predicted_BFA1000, linetype = "Ensemble trend"),
      color = "#2b2b2b", size = 1.2, alpha = 0.3,
      inherit.aes = FALSE
    ) +
    
    # Reference line
    geom_hline(
      aes(yintercept = BFA1000_1991_2024, linetype = "Observed (1991–2024)"),
      data = ens_scen |> filter(Period == "1981-2010"),
      color = "#2b2b2b",
      size = 0.8
    ) +
    
    # Model points on top
    geom_point(size = 3, alpha = 0.8) +
    
    # X axis scale
    scale_x_continuous(
      breaks = x_labels,
      labels = x_labels,
      limits = c(1988, 2092)
    ) +
    
    # Color legend for models
    scale_color_manual(
      values = custom_colors_GCM,
      breaks = c("Baseline", "cmcc-esm2", "ec-earth3", "gfdl-esm4",
                 "mpi-esm1-2-hr", "mri-esm2-0", "taiesm1"),
      name = "Climate data",
      guide = guide_legend(order = 1)
    ) +
    
    # Linetype legend for both lines
    scale_linetype_manual(
      values = c("Observed (1991–2024)" = "dashed", "Ensemble trend" = "solid"),
      # values = c("Ensemble trend" = "solid"),
      name = NULL,
      guide = guide_legend(
        override.aes = list(
          color = "#2b2b2b",
          size = c(0.8, 1.2),
          alpha = c(0.3, 1)
          # size = c(0.8),
          # alpha = c(0.3)
        ),
        order = 2,
        keywidth = 1.8
      )
    ) +
    
    # Labels and theme
    labs(
      title = "Ensemble mean model of BFAI",
      x = "Year",
      y = "Predicted BFAI"
    ) +
    
    theme_bw() +
    theme(
      legend.position = "right",
      legend.text = element_text(size = 9),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.title.x = element_text(margin = margin(t = 15)),
      strip.text = element_text(face = "bold")
    )
  
  # Save the plot
  if (use_country == TRUE) {
    plot_name <- paste0(out_path, "/scenarios/scenarios_", country_code, "_", ssp_use, "_all_models_predictions_for_scenarios_ensemble_mean_country.png")
  } else {
    plot_name <- paste0(out_path, "/scenarios/scenarios_", country_code, "_", ssp_use, "_all_models_predictions_for_scenarios_ensemble_mean.png")
  }
  ggsave(plot_name, plot = scenario_p, width = 140, height = 100, dpi = 600, units = 'mm', device = ragg::agg_png)
}

# countries <- calibration_data |>
#   distinct(Country) |>
#   pull(Country)
# 
# ssps <- c("ssp126", "ssp245", "ssp370", "ssp585")
# 
# # To speed up
# # ssps <-"ssp245"
# countries <- c("CZE", "ESP", "GRC", "ITA", "PRT", "ROU", "SWE")
# # countries <- c("AUT", "BGR", "CZE", "ESP", "FRA", "GRC", "ITA", "PRT", "ROU", "SWE")
# 
# for (country_code in countries) {
#   for (ssp_use in ssps) {
#     make_country_scenario_plots(country_code, ssp_use)
#   }
# }

# purrr::walk(scenario_settings$countries, \(country_code) {
#   purrr::walk(scenario_settings$ssps, \(ssp_use) {
#     make_country_scenario_plots(country_code, ssp_use)
#   })
# })

options(future.globals.maxSize = 3 * 1024^3)

log_message("Parallel ON for scenario plotting")

future::plan(multisession, workers = num_cores_plot)
gc()

furrr::future_walk(
  scenario_settings$countries,
  \(country_code) {
    purrr::walk(scenario_settings$ssps, \(ssp_use) {
      make_country_scenario_plots(country_code, ssp_use)
    })
  },
  .options = furrr::furrr_options(
    seed = TRUE,
    packages = "tidyverse"
  )
)

future::plan(sequential)
gc()

log_message("Parallel OFF after scenario plotting")