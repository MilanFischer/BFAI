# ============================================================
# Perturbation audit for scenario robustness
# ============================================================

audit_model_perturbation <- function(
    final_model,
    base_data,
    predictors,
    n_rows_per_country = 5,
    n_steps = 25,
    max_multiplier = 2.0,
    max_log_increase = log(30),
    max_step_jump = log(3),
    min_monotonic_share = 0.70,
    seed = 123
) {
  
  set.seed(seed)
  
  audit_base <- base_data |>
    dplyr::group_by(Country) |>
    dplyr::mutate(.rand = stats::runif(dplyr::n())) |>
    dplyr::arrange(.rand, .by_group = TRUE) |>
    dplyr::slice_head(n = n_rows_per_country) |>
    dplyr::ungroup() |>
    dplyr::select(-.rand) |>
    dplyr::mutate(.audit_id = dplyr::row_number())
  
  hazard_vars <- predictors[stringr::str_detect(
    predictors,
    "FWI|VPD|DFMC10H_u|AWD|AWP|AWR.*u"
  )]
  
  moisture_vars <- predictors[stringr::str_detect(
    predictors,
    "^DFMC10H($|_)|^AWR($|_)"
  )]
  
  if (length(hazard_vars) == 0) {
    return(tibble::tibble(
      status = "fail",
      reason = "no_hazard_variables_found",
      max_jump = NA_real_,
      max_increase = NA_real_,
      monotonic_share = NA_real_
    ))
  }
  
  stress_data <- purrr::map_dfr(
    seq(1, max_multiplier, length.out = n_steps),
    function(m) {
      
      tmp <- audit_base
      
      for (v in hazard_vars) {
        if (v %in% names(tmp) && is.numeric(tmp[[v]])) {
          tmp[[v]] <- tmp[[v]] * m
        }
      }
      
      for (v in moisture_vars) {
        if (v %in% names(tmp) && is.numeric(tmp[[v]])) {
          tmp[[v]] <- tmp[[v]] / m
        }
      }
      
      tmp |> dplyr::mutate(.stress = m)
    }
  )
  
  pred <- predict(final_model, stress_data) |>
    dplyr::bind_cols(stress_data |> dplyr::select(.audit_id, Country, .stress)) |>
    dplyr::arrange(.audit_id, .stress) |>
    dplyr::group_by(.audit_id) |>
    dplyr::mutate(
      step_diff = .pred - dplyr::lag(.pred),
      increase_from_start = .pred - dplyr::first(.pred)
    ) |>
    dplyr::ungroup()
  
  pred |>
    dplyr::summarise(
      max_jump = max(abs(step_diff), na.rm = TRUE),
      max_increase = max(increase_from_start, na.rm = TRUE),
      monotonic_share = mean(step_diff >= -0.05, na.rm = TRUE)
    ) |>
    dplyr::mutate(
      status = dplyr::if_else(
        max_jump <= max_step_jump &
          max_increase <= max_log_increase &
          monotonic_share >= min_monotonic_share,
        "pass",
        "fail"
      ),
      reason = dplyr::case_when(
        max_jump > max_step_jump ~ "large_step_jump",
        max_increase > max_log_increase ~ "explosive_growth",
        monotonic_share < min_monotonic_share ~ "non_monotonic_response",
        TRUE ~ "ok"
      )
    )
}


run_perturbation_audit_for_stacking <- function(
    race_results,
    matched_results,
    calibration_data,
    predictors,
    out_path,
    settings,
    scenario_audit_data = NULL,
    seed = 123
) {
  
  audit_all <- list()
  keep_configs_by_model <- list()
  
  # Scenario exceedance audit is optional. It is applied only when:
  #   1) scenario_audit_data is supplied, and
  #   2) settings$max_allowed_scenario_BFAI is supplied.
  #
  # This preserves the previous functionality when these inputs are not used.
  use_scenario_audit <- !is.null(scenario_audit_data) &&
    !is.null(settings$max_allowed_scenario_BFAI)
  
  for (i in seq_len(nrow(matched_results))) {
    
    model_id <- matched_results$wflow_id[i]
    
    message("Candidate-level perturbation audit: ", model_id)
    
    tune_res <- workflowsets::extract_workflow_set_result(race_results, model_id)
    wf <- workflowsets::extract_workflow(race_results, model_id)
    
    candidate_grid <- tune::collect_metrics(tune_res) |>
      dplyr::filter(.metric == "rmse") |>
      dplyr::select(
        -dplyr::any_of(c(".metric", ".estimator", "mean", "n", "std_err"))
      ) |>
      dplyr::distinct(.config, .keep_all = TRUE)
    
    model_audit <- purrr::map_dfr(
      seq_len(nrow(candidate_grid)),
      function(j) {
        
        cfg <- candidate_grid[j, ]
        
        message("  auditing ", model_id, " | ", cfg$.config)
        
        params <- cfg |> dplyr::select(-.config)
        
        set.seed(seed)
        
        final_model <- wf |>
          tune::finalize_workflow(params) |>
          workflows::fit(data = calibration_data)
        
        if (isTRUE(settings$use_synthetic_audit)) {
          
          audit <- audit_model_perturbation(
            final_model = final_model,
            base_data = calibration_data,
            predictors = predictors,
            n_rows_per_country = settings$n_rows_per_country,
            n_steps = settings$n_steps,
            max_multiplier = settings$max_multiplier,
            max_log_increase = settings$max_log_increase,
            max_step_jump = settings$max_step_jump,
            min_monotonic_share = settings$min_monotonic_share,
            seed = seed
          )
          
        } else {
          
          audit <- tibble::tibble(
            status = "pass",
            reason = "synthetic_audit_skipped",
            max_jump = NA_real_,
            max_increase = NA_real_,
            monotonic_share = NA_real_
          )
        }
        
        if (use_scenario_audit) {
          
          scen_pred <- predict(final_model, new_data = scenario_audit_data) |>
            dplyr::mutate(predicted_BFA1000 = exp(.pred))
          
          q_scen_bfai <- suppressWarnings(
            quantile(
              scen_pred$predicted_BFA1000,
              probs = settings$scenario_bfai_q,
              na.rm = TRUE
            )
          )
          
          max_scen_bfai <- suppressWarnings(
            max(scen_pred$predicted_BFA1000, na.rm = TRUE)
          )
          
          q_scen_bfai <- as.numeric(q_scen_bfai)
          
          if (is.infinite(max_scen_bfai) && max_scen_bfai < 0) {
            max_scen_bfai <- NA_real_
          }
          
          audit <- audit |>
            dplyr::mutate(
              scenario_q_BFAI = q_scen_bfai,
              scenario_q_used = settings$scenario_bfai_q,
              max_scenario_BFAI = max_scen_bfai,
              max_allowed_scenario_BFAI = settings$max_allowed_scenario_BFAI,
              status = dplyr::if_else(
                status == "pass" &
                  is.finite(q_scen_bfai) &
                  q_scen_bfai <= settings$max_allowed_scenario_BFAI,
                "pass",
                "fail"
              ),
              reason = dplyr::case_when(
                !is.finite(q_scen_bfai) ~ "non_finite_scenario_prediction",
                q_scen_bfai > settings$max_allowed_scenario_BFAI ~
                  "scenario_BFAI_q_exceedance",
                TRUE ~ reason
              )
            )
          
        } else {
          
          audit <- audit |>
            dplyr::mutate(
              scenario_q_BFAI = NA_real_,
              scenario_q_used = NA_real_,
              max_scenario_BFAI = NA_real_,
              max_allowed_scenario_BFAI = NA_real_
            )
        }
        
        audit |>
          dplyr::mutate(
            model = model_id,
            .config = cfg$.config
          )
      }
    )
    
    audit_all[[model_id]] <- model_audit
    
    keep_configs_by_model[[model_id]] <- model_audit |>
      dplyr::filter(status == "pass") |>
      dplyr::pull(.config)
  }
  
  robustness_table <- dplyr::bind_rows(audit_all) |>
    dplyr::relocate(model, .config) |>
    dplyr::mutate(
      instability_score =
        as.numeric(scale(max_jump)) +
        2 * as.numeric(scale(max_increase)) +
        as.numeric(scale(1 - monotonic_share))
    )
  
  readr::write_csv(
    robustness_table,
    file.path(out_path, "candidate_perturbation_robustness.csv")
  )
  
  race_results_for_stack <- race_results
  
  for (model_id in names(keep_configs_by_model)) {
    
    keep_configs <- keep_configs_by_model[[model_id]]
    
    if (length(keep_configs) == 0) {
      warning("No candidate configurations passed for ", model_id)
      next
    }
    
    row_id <- which(race_results_for_stack$wflow_id == model_id)
    
    tune_res <- workflowsets::extract_workflow_set_result(
      race_results_for_stack,
      model_id
    )
    
    tune_res$.metrics <- purrr::map(
      tune_res$.metrics,
      ~ dplyr::filter(.x, .config %in% keep_configs)
    )
    
    if (".predictions" %in% names(tune_res)) {
      tune_res$.predictions <- purrr::map(
        tune_res$.predictions,
        ~ dplyr::filter(.x, .config %in% keep_configs)
      )
    }
    
    if (".extracts" %in% names(tune_res)) {
      tune_res$.extracts <- purrr::map(
        tune_res$.extracts,
        ~ dplyr::filter(.x, .config %in% keep_configs)
      )
    }
    
    race_results_for_stack$result[[row_id]] <- tune_res
  }
  
  robust_model_ids <- names(keep_configs_by_model)[
    purrr::map_int(keep_configs_by_model, length) > 0
  ]
  
  race_results_for_stack <- race_results_for_stack[
    race_results_for_stack$wflow_id %in% robust_model_ids,
  ]
  
  class(race_results_for_stack) <- class(race_results)
  
  matched_results_for_scenarios <- matched_results |>
    dplyr::filter(wflow_id %in% robust_model_ids)
  
  return(list(
    race_results_for_stack = race_results_for_stack,
    robustness_table = robustness_table,
    robust_model_ids = robust_model_ids,
    matched_results_for_scenarios = matched_results_for_scenarios,
    keep_configs_by_model = keep_configs_by_model
  ))
}

################################################################################
prepare_scenario_audit_data <- function(
    scenario_root = "./inputs",
    countries,
    clean_data,
    scenario = "ssp585"
) {
  
  purrr::map_dfr(countries, function(country_code) {
    
    scenario_country_files <- list.files(
      path = scenario_root,
      pattern = paste0(country_code, "_forest\\.csv$"),
      recursive = TRUE,
      full.names = TRUE
    ) |>
      purrr::keep(\(x) stringr::str_detect(x, "scenariomip")) |>
      purrr::keep(\(x) stringr::str_detect(x, scenario))
    
    country_static <- clean_data |>
      dplyr::filter(Country == country_code) |>
      dplyr::summarise(
        Pines = mean(Pines, na.rm = TRUE),
        Conifers = mean(Conifers, na.rm = TRUE),
        Broadleaved = mean(Broadleaved, na.rm = TRUE)
      )
    
    scenario_country_files |>
      purrr::map_dfr(function(file) {
        
        path_parts <- stringr::str_split(file, "/", simplify = TRUE)
        
        readr::read_csv(file, show_col_types = FALSE) |>
          tidyr::pivot_longer(
            cols = dplyr::matches("^\\d{4}$"),
            names_to = "Year",
            values_to = "value"
          ) |>
          dplyr::mutate(
            Country = country_code,
            Scenario = stringr::str_extract(file, "ssp\\d+"),
            GCM = path_parts[length(path_parts) - 2],
            Period = as.numeric(path_parts[length(path_parts) - 1]),
            Year = as.numeric(Year),
            varname = paste(Variable, Season, sep = "_")
          ) |>
          dplyr::select(Country, Scenario, GCM, Period, Year, varname, value)
      }) |>
      tidyr::pivot_wider(names_from = varname, values_from = value) |>
      dplyr::mutate(
        Pines = country_static$Pines,
        Conifers = country_static$Conifers,
        Broadleaved = country_static$Broadleaved
      )
  }) |>
    dplyr::rename_with(
      \(x) x |>
        stringr::str_replace_all("\\+", "_plus") |>
        stringr::str_replace_all("-", "_")
    )
}