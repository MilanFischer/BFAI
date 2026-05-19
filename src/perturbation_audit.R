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
      
      tmp |>
        dplyr::mutate(.stress = m)
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
    seed = 123
) {
  
  robustness_results <- list()
  
  for (i in seq_len(nrow(matched_results))) {
    
    model_id <- matched_results$wflow_id[i]
    
    message("Perturbation audit: ", model_id)
    
    best_results <- race_results |>
      workflowsets::extract_workflow_set_result(model_id) |>
      tune::select_best(metric = "rmse")
    
    final_model <- race_results |>
      workflowsets::extract_workflow(model_id) |>
      tune::finalize_workflow(best_results) |>
      parsnip::fit(data = calibration_data)
    
    robustness_results[[model_id]] <- audit_model_perturbation(
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
    ) |>
      dplyr::mutate(model = model_id)
  }
  
  robustness_table <- dplyr::bind_rows(robustness_results) |>
    dplyr::relocate(model) |>
    dplyr::mutate(
      instability_score =
        as.numeric(scale(max_jump)) +
        as.numeric(scale(max_increase))
    )
  
  readr::write_csv(
    robustness_table,
    file.path(out_path, "model_perturbation_robustness.csv")
  )
  
  print(robustness_table)
  
  # ------------------------------------------------
  # Keep only most stable passing models
  # ------------------------------------------------
  
  n_keep_stable_models <- settings$n_keep_stable_models
  
  if (is.null(n_keep_stable_models)) {
    n_keep_stable_models <- Inf
  }
  
  robust_model_ids <- robustness_table |>
    dplyr::filter(status == "pass") |>
    dplyr::arrange(instability_score) |>
    dplyr::slice_head(n = n_keep_stable_models) |>
    dplyr::pull(model)
  
  message("Models passing perturbation audit: ", length(robust_model_ids))
  
  if (length(robust_model_ids) == 0) {
    warning(
      "No models passed perturbation audit. Falling back to unfiltered race_results."
    )
    return(race_results)
  }
  
  race_results_for_stack <- race_results |>
    dplyr::filter(wflow_id %in% robust_model_ids)
  
  return(list(
    race_results_for_stack = race_results_for_stack,
    robustness_table = robustness_table,
    robust_model_ids = robust_model_ids
  ))
}