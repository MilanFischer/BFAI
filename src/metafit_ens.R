# Metafit
# Extract CV predictions and collapse by row
extract_preds <- function(wflow_set, id) {
  wf_result <- wflow_set |>
    filter(wflow_id == id) |>
    pull(result)
  
  preds <- wf_result[[1]] |> collect_predictions()
  
  outcome_col <- setdiff(names(preds), c(".pred", ".row", ".config", "id", "Resample", "hidden_units", "penalty", "epochs"))
  
  preds |>
    group_by(.row) |>
    summarise(
      !!id := mean(.pred, na.rm = TRUE),
      .truth = mean(!!sym(outcome_col[1]), na.rm = TRUE),
      .groups = "drop"
    )
}

metafit_ens <- function(race_results,
                            calibration_data,
                            verification_data,
                            rmse_deviation = 1.05,
                            model = "lm"){
  
  # Model check
  if (!model %in% c("lm", "glm", "xgboost", "nnet", "glmnet", "ranger", "cubist", "bart")) {
    stop("Model must be one of: 'lm', 'glm', 'xgboost', 'nnet', 'glmnet', 'ranger', 'cubist', or 'bart'")
  }
  
  # Step 1: Select top N models based on RMSE
  top_models <- race_results |>
    rank_results() |>
    filter(.metric == "rmse") |>
    group_by(wflow_id) |>
    summarize(mean_rmse = mean(mean), .groups = "drop") |>
    arrange(mean_rmse)
  
  # Automatically include all models within certain deviation from the best RMSE
  best_rmse <- min(top_models$mean_rmse)
  threshold <- best_rmse * rmse_deviation  # e.g. 1.05 is 5% within best
  
  top_n_models <- top_models |>
    filter(mean_rmse <= threshold) |>
    pull(wflow_id)
  
  # Step 2: Create stacked prediction dataset for meta-learner training
  pred_dfs <- lapply(top_n_models, extract_preds, wflow_set = race_results)
  
  stack_preds <- reduce(
    pred_dfs,
    ~ left_join(.x, .y |> select(-.truth), by = ".row")
  )
  
  # Step 3: Fit meta-learner (you can swap for xgboost, etc.)
  meta_fit <- switch(model,
                     # Linear Regression (base model)
                     "lm" = linear_reg() |> set_engine("lm"),
                     # Generalized Linear Model
                     "glm" = linear_reg() |> set_engine("glm"),
                     # XGBoost (gradient boosting trees)
                     "xgboost" = boost_tree(trees = 100, learn_rate = 0.1) |> set_engine("xgboost") |> set_mode("regression"),
                     # Neural Network (MLP using nnet)
                     "nnet" = mlp(hidden_units = 5, penalty = 0.001, epochs = 100) |> set_engine("nnet") |> set_mode("regression"),
                     # Regularized Linear Model (Elastic Net via glmnet)
                     "glmnet" = linear_reg(penalty = 0.01, mixture = 0.5) |> set_engine("glmnet"),
                     # Random Forest (via ranger)
                     "ranger" = rand_forest(trees = 500) |>  set_engine("ranger") |> set_mode("regression"),
                     # Rule-Based Ensemble Model (Cubist)
                     "cubist" = cubist_rules(committees = 3) |> set_engine("Cubist") |> set_mode("regression"),
                     # Bayesian Additive Regression Trees (BART)
                     "bart" = bart() |> set_engine("dbarts") |> set_mode("regression")
                     
  ) |> fit(.truth ~ ., data = stack_preds |> select(-.row))

  # Step 4: Evaluate on CV resamples (optional)
  stack_preds <- stack_preds |> 
    mutate(ensemble_pred = predict(meta_fit, new_data = stack_preds |> select(-.row, -.truth))[[1]])
  
  rmse(stack_preds, truth = .truth, estimate = ensemble_pred)
  rsq(stack_preds, truth = .truth, estimate = ensemble_pred)
  
  # Step 5: Finalize and fit best models on full calibration data
  fitted_workflows <- setNames(vector("list", length(top_n_models)), top_n_models)
  
  for (id in top_n_models) {
    wf_result_raw <- race_results |>
      filter(wflow_id == id) |>
      pull(result)
    
    wf_result <- wf_result_raw[[1]]  # tuning result (e.g., from tune_race_anova)
    
    # Correct way to get best params directly
    best_params <- select_best(wf_result, metric = "rmse")
    
    # Extract and finalize the workflow
    wf <- extract_workflow(wf_result)
    finalized_wf <- finalize_workflow(wf, best_params)
    
    # Fit the finalized workflow on the full calibration set
    fitted_workflows[[id]] <- fit(finalized_wf, data = calibration_data)
  }
  
  # Step 6: Define ensemble object and prediction function
  custom_ensemble <- list(
    model = meta_fit,
    model_ids = top_n_models,
    base_models = fitted_workflows,
    
    predict = function(new_data) {
      base_preds <- lapply(top_n_models, function(id) {
        predict(fitted_workflows[[id]], new_data = new_data) |> rename(!!id := .pred)
      }) |> reduce(bind_cols)
      
      # return prediction as a tibble (unchanged)
      predict(meta_fit, new_data = base_preds)
    }
  )
  
  class(custom_ensemble) <- "custom_ensemble"
  
  # Step 7: Predict on calibration and verification sets
  ens_calibration_pred <- custom_ensemble$predict(calibration_data) |> 
    bind_cols(calibration_data) |> 
    mutate(dataset = "calibration")
  
  ens_verification_pred <- custom_ensemble$predict(verification_data) |> 
    bind_cols(verification_data) |> 
    mutate(dataset = "verification")
  
  # Step 8: Evaluate on both sets
  reg_metrics <- metric_set(rmse, rsq)
  
  ens_calibration_pred |> reg_metrics(log_BFA1000, .pred)
  ens_verification_pred |> reg_metrics(log_BFA1000, .pred)
  
  # Step 9: Combine for downstream
  ens_model_pred_df <- bind_rows(ens_calibration_pred, ens_verification_pred)
  
  return(list(
    predictions = ens_model_pred_df,
    ensemble = custom_ensemble
  ))
}

predict.custom_ensemble <- function(object, new_data, ...) {
  object$predict(new_data)
}