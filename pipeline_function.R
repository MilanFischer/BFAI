run_pipeline <- function(run_i, runs) {
  
  on.exit({
    future::plan(sequential)
    gc()
  }, add = TRUE)
  
  cfg <- runs[run_i, ]
  
  source("./src/log_message.R")
  
  log_message("Loading libraries and sourcing functions")
  source("./src/load_libraries.R")
  source("./src/colors.R")
  source("./src/model_specification.R")
  source("./src/corellation_filter.R")
  source("./src/boruta.R")
  source("./src/perturbation_audit.R")
  source("./src/metafit_ens.R")
  source("./src/files_manage.R")
  
  target                    <- "log_BFA1000"
  run_ID                    <- cfg$run_ID
  use_country               <- cfg$use_country
  use_meteo                 <- cfg$use_meteo
  use_winter                <- cfg$use_winter
  use_year                  <- cfg$use_year
  metamodel                 <- cfg$metamodel
  cor_thresh                <- cfg$cor_thresh
  use_perturbation_audit    <- cfg$use_perturbation_audit
  use_synthetic_audit       <- cfg$use_synthetic_audit
  use_scenario_audit        <- cfg$use_scenario_audit 
  scenario_bfai_multiplier  <- cfg$scenario_bfai_multiplier
  grid_ini                  <- cfg$grid_ini
  grid_race                 <- cfg$grid_race
  n_ens_reps                <- cfg$n_ens_reps
  num_cores_tune            <- cfg$num_cores_tune
  num_cores_plot            <- cfg$num_cores_plot
  out_path                  <- cfg$out_path
  
  log_message("Running ", basename(out_path),
          " | cor_thresh = ", cor_thresh,
          " | use_country = ", use_country,
          " | use_meteo = ", use_meteo)
  
  dir.create(file.path(out_path, "scenarios"), recursive = TRUE, showWarnings = FALSE)
  
  perturbation_audit_settings <- list(
    use_synthetic_audit = use_synthetic_audit,
    use_scenario_audit = use_scenario_audit,
    n_rows_per_country = 5,
    n_steps = 25,
    max_multiplier = 2.0,
    max_log_increase = log(20),
    max_step_jump = log(3),
    min_monotonic_share = 0.80,
    # Maximum allowed exceedance relative to historical high-end BFAI
    # during SSP585 scenario screening.
    scenario_bfai_q = 0.99,
    scenario_bfai_multiplier = scenario_bfai_multiplier,
    seed = 123
  )
  
  scenario_settings <- list(
    countries = c("AUT", "BEL", "BGR", "CHE", "CZE", "DEU", "DNK", "ESP", "FIN", "FRA", "GRC", "HRV", "ITA", "LTU", "LVA", "NOR", "POL", "PRT", "ROU", "SVK", "SVN", "SWE"),
    ssps = c("ssp126", "ssp245", "ssp370", "ssp585")
  )
  
  # Separate calibration and verification datasets
  calibration_years <- seq(1990, 2024)[(seq(1990, 2024) - 1990) %% 3 != 2]  # 1st and 2nd years in each 3-year block
  verification_years <- seq(1990, 2024)[(seq(1990, 2024) - 1990) %% 3 == 2]  # 3rd year in each 3-year block
  
  # Set up parallel processing (to speed up computations)
  registerDoFuture()
  future::plan(sequential)

  #------------------------------------
  # Load data and handle missing values
  
  # Path to baseline folder
  baseline_path <- "./inputs/baseline"
  
  # Get only *_forest.csv files
  forest_files <- list.files(
    path = baseline_path,
    pattern = "_forest\\.csv$",
    full.names = TRUE
  )
  
  data <- forest_files |> 
    map_dfr(~ {
      read_csv(.x, show_col_types = FALSE) |> 
        pivot_longer(
          cols = matches("^\\d{4}$"),
          names_to = "Year",
          values_to = "value"
        ) |> 
        mutate(
          Country = str_extract(basename(.x), "^[A-Z]{3}"),
          Year = as.numeric(Year),
          varname = paste(Variable, Season, sep = "_")
        ) |> 
        select(Country, Year, varname, value)
    }) |> 
    pivot_wider(
      names_from = varname,
      values_from = value
    ) |> 
    filter(Year >= 1990) |> 
    arrange(Country, Year)
  
  BFAI_data <- read_csv("./inputs/BFAI_1990-2024.csv", show_col_types = FALSE)
  
  data <- data |>
    left_join(BFAI_data, by = c("Country", "Year"))
  
  # Country for which no BFAI is not available: "EST", "GBR", "HUN", "IRL", "NLD"
  data <- data |>
    filter(!Country %in% c("EST", "GBR", "HUN", "IRL", "NLD"))
  
  # Replace infinite values with NA and drop them
  clean_data <- data |> 
    mutate(across(where(is.numeric), ~na_if(., Inf))) |>  # Replace Inf with NA
    drop_na()
  
  # Remove unused
  clean_data <- clean_data |> 
    select(-c("FA_kha", "BFA_ha"))
  
  if(use_meteo == FALSE){
    clean_data <- clean_data |>
      select(-starts_with(c("TMAX_", "TMIN_", "TAVG_", "RH_", "RHmin", "PREC", "SRAD", "WIND")))
  }
  
  if(use_winter == FALSE){
    clean_data <- clean_data |>
      select(-matches("ONDJFM|NDJF"))
  }
  
  clean_data <- clean_data |> 
    rename(
      BFA1000 = BFAI,
      Conifers = Share_Con,
      Pines    = Share_Pines
    ) |> 
    mutate(
      Broadleaved = 1 - Conifers
    )
  
  clean_data <- clean_data |> 
    relocate(BFA1000, .after = Year)
  
  clean_data <- clean_data |> 
    rename_with(~ gsub("\\+", "_plus", .))
  
  clean_data <- clean_data |> 
    rename_with(~ gsub("\\-", "_", .))
  
  # if(use_country == TRUE){
  #   clean_data <- clean_data |> select(-c("Pines", "Conifers", "Broadleaved"))
  # }
  
  # Derive the original variable name from the target
  original_var <- sub("^log_", "", target)
  
  # Apply transformation using dynamic variable names
  clean_data <- clean_data |> 
    mutate(!!target := log(.data[[original_var]])) |> 
    select(Country, Year, all_of(target), everything(), -all_of(original_var))
  
  calibration_data <- clean_data |>
    filter(Year %in% calibration_years)
  
  verification_data <- clean_data |>
    filter(Year %in% verification_years)
  
  config_path <- file.path(out_path, "config.yml")
  
  if (file.exists(config_path)) {
    config <- yaml::read_yaml(config_path)
    
    # Assign to global env or current env as needed
    predictors <- config$predictors
    
    use_country <- config$use_country
    
    use_meteo <- config$use_meteo 
    
    use_winter <- config$use_winter
    
    use_year <- config$use_year
    
    target <- config$target
    
    cor_thresh <- config$cor_thresh
    
    metamodel <- config$metamodel
    
    use_perturbation_audit <- config$use_perturbation_audit
    
    use_synthetic_audit <- config$use_synthetic_audit
    
    use_scenario_audit <- config$use_scenario_audit
    
    scenario_bfai_multiplier <- config$scenario_bfai_multiplier
    
    grid_ini <- config$grid_ini
    
    grid_race <- config$grid_race
    
    n_ens_reps <- config$n_ens_reps
    
    num_cores_tune <- config$num_cores_tune
    
    num_cores_plot <- config$num_cores_plot
    
    seed <- config$seed
    
    log_message("Loaded config from: ", config_path)
  } else {
    
    log_message("No config.yml found. Generating new config.")
    
    seed <- 1234
    
    # Remove highly correlated predictors
    log_message("Starting correlation filter")
    cor_filtered_cols <- corr_filter(
      data = calibration_data,
      target = target,
      cutoff = cor_thresh,
      use_year = use_year
    )
    
    log_message("Starting Boruta")
    boruta_filtered_cols <- get_boruta_predictors(calibration_data |> select(all_of(cor_filtered_cols)),
                                                  target = target, use_country = use_country, use_year = use_year, min_freq = 5, v = 10, seed = seed)
    
    predictors_only <- setdiff(boruta_filtered_cols, c(target, if (!use_year) "Year"))
    
    # RFE with correct predictors
    log_message("Starting RFE")
    ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
    rfe_result <- rfe(calibration_data[, predictors_only], calibration_data[[target]],
                      sizes = length(predictors_only), rfeControl = ctrl)
    
    # Extract variable importance
    vip_df <- data.frame(Predictors = row.names(varImp(rfe_result)),
                         Importance = varImp(rfe_result)$Overall)
    
    # Sort by importance
    vip_df <- vip_df[order(-vip_df$Importance), ]
    
    # Define plot name
    plot_name <- paste0(out_path, "/feature_importance.png")
    
    # Create plot
    p <- ggplot(vip_df, aes(x = reorder(Predictors, Importance), y = Importance)) +
      geom_col() +
      coord_flip() +
      labs(title = "Variable Importance (RFE - Random Forest)",
           x = "Variable", y = "Importance") +
      theme_bw()
    
    # Save plot
    ggsave(plot_name, plot = p, width = 8, height = 6, dpi = 300, device = ragg::agg_png)
    
    final_cols <- c(target, vip_df$Predictors)
    
    predictors <- setdiff(final_cols, target)
    
    # Save predictors and seed to config.yml
    log_message("Saving config")
    config <- list(
      predictors = predictors,
      use_country = use_country,
      use_meteo = use_meteo,
      use_winter = use_winter,
      use_year = use_year,
      target = target,
      cor_thresh = cor_thresh,
      metamodel = metamodel,
      use_perturbation_audit = use_perturbation_audit,
      use_synthetic_audit = use_synthetic_audit,
      use_scenario_audit = use_scenario_audit,
      scenario_bfai_multiplier = scenario_bfai_multiplier,
      grid_ini = grid_ini,
      grid_race = grid_race,
      n_ens_reps = n_ens_reps,
      num_cores_tune = num_cores_tune,
      num_cores_plot = num_cores_plot,
      seed = seed
    )
    
    write_yaml(config, config_path)
    log_message("Saved new config to: ", config_path)
  }
  
  needed_cols <- unique(c("Country", "Year", target, predictors))
  
  calibration_data <- calibration_data |>
    select(any_of(needed_cols))
  
  verification_data <- verification_data |>
    select(any_of(needed_cols))
  
  
  # Create a formula dynamically
  formula <- as.formula(paste(target, "~", paste(predictors, collapse = " + ")))
  
  if (use_country == TRUE) {
    no_pre_proc_rec <- recipe(formula, data = calibration_data) |>
      step_nzv(all_predictors()) |>
      step_lencode_glm(Country, outcome = target)
    
  } else {
    no_pre_proc_rec <- recipe(formula, data = calibration_data) |>
      step_nzv(all_predictors())
  }
  
  ###############################################################
  
  # Cross-validation folds
  set.seed(seed)
  cv_folds <- vfold_cv(calibration_data, v = 10, strata = all_of(target))
  
  # Extend the recipe to normalize numeric predictors
  normalized_rec <- no_pre_proc_rec |> 
    step_normalize(all_numeric_predictors())
  
  poly_rec <-
    normalized_rec |>
    step_poly(all_numeric_predictors()) |>  # Exclude dummy variables
    step_interact(~ all_numeric_predictors():all_numeric_predictors())
  
  # Define a set of workflows without preprocessing
  no_pre_proc <- workflow_set(
    preproc = list(simple = no_pre_proc_rec),
    models = list(
      `Multivariate Adaptive Regression Splines` = mars_spec,
      `Classification and Regression Trees` = cart_spec,
      `Bagged Classification and Regression Trees` = bag_cart_spec,
      `Random Forest` = rf_spec,
      `Extreme Gradient Boosting` = xgb_spec,
      `Cubist Regression` = cubist_spec,
      `Bayesian Additive Regression Trees` = bart_spec,
      `Generalized Linear Model` = glm_spec
    )
  )
  
  # Define a set of workflows with normalization
  normalized <- workflow_set(
    preproc = list(normalized = normalized_rec),
    models = list(
      `Support Vector Machine (Radial)` = svm_r_spec,
      `Support Vector Machine (Polynomial)` = svm_p_spec,
      `K-Nearest Neighbors (Normalized)` = knn_spec,
      `Artificial Neural Network` = nnet_spec,
      `Support Vector Machine (Linear)` = svm_linear_spec
    )
  )
  
  # Define a set of workflows with polynomial and interaction features
  poly_models <- list(
    `Elastic Net Regression` = linear_reg_spec,
    `K-Nearest Neighbors (Poly Features)` = knn_spec
  )
  
  # Create workflow set
  with_features <- workflow_set(
    preproc = list(full_quad = poly_rec),
    models = poly_models
  )
  
  # Manually assign desired workflow IDs
  with_features$wflow_id <- names(poly_models)
  
  # Combine all workflow sets into a single set
  all_workflows <- 
    bind_rows(no_pre_proc, normalized, with_features) |>  
    # Make the workflow ID's a little more simple: 
    mutate(wflow_id = gsub("(simple_)|(normalized_)", "", wflow_id))
  
  # Display the combined workflow set
  all_workflows
  
  # Be careful here
  # all_workflows <- all_workflows |> 
  #   slice(3:4)
  
  # Remove any rows where wflow_id contains "KNN"
  all_workflows <- all_workflows |> 
    filter(!grepl("K-Nearest Neighbors", wflow_id))
  
  # Remove CART
  all_workflows <- all_workflows |> 
    filter(wflow_id != "Classification and Regression Trees")
  
  # Final selection
  print(all_workflows)
  
  grid_ctrl <-
    control_grid(
      save_pred = TRUE,
      parallel_over = "resamples", # "everything"
      save_workflow = TRUE,
      allow_par = TRUE
    )
  
  log_message("Parallel ON for grid tuning")
  future::plan(multisession, workers = num_cores_tune)
  gc()
  
  set.seed(seed)
  # Run the grid search
  grid_results <-
    all_workflows |>
    workflow_map(
      seed = seed,
      resamples = cv_folds,
      grid = grid_ini, # Reduced grid size for debugging, otherwise set >50
      control = grid_ctrl
    )
  
  log_message("Parallel OFF after grid tuning")
  future::plan(sequential)
  gc()
  
  # Verify results and notes again
  show_notes(grid_results)
  # grid_results |>  
  #   rank_results() |>  
  #   filter(.metric == "rmse") |>  
  #   select(model, .config, rmse = mean, rank) |> View()
  
  # autoplot(
  #   grid_results,
  #   rank_metric = "rmse",  # <- how to order models
  #   metric = "rmse",       # <- which metric to visualize
  #   select_best = TRUE     # <- one point per workflow
  # ) +
  #   geom_text(aes(y = mean - 0.2, label = wflow_id), angle = 90, hjust = 1) +
  #   lims(y = c(0.6, 1.4)) +
  #   theme(legend.position = "none")
  
  race_ctrl <-
    control_race(
      save_pred = TRUE,
      parallel_over = "resamples", # "everything"
      save_workflow = TRUE,
      allow_par = TRUE
    )
  
  log_message("Parallel ON for racing")
  future::plan(multisession, workers = num_cores_tune)
  gc()
  
  set.seed(seed)
  race_results <-
    all_workflows |>
    workflow_map(
      "tune_race_anova",
      seed = seed,
      resamples = cv_folds,
      grid = grid_race, # e.g. 100
      control = race_ctrl
    )
  
  log_message("Parallel OFF after racing")
  future::plan(sequential)
  gc()
  
  set.seed(seed)
  # race_results |>  
  #   rank_results() |>  
  #   filter(.metric == "rmse") |>  
  #   select(model, .config, rmse = mean, rank) |> View()
  # 
  # autoplot(
  #   race_results,
  #   rank_metric = "rmse",  # <- how to order models
  #   metric = "rmse",       # <- which metric to visualize
  #   select_best = TRUE     # <- one point per workflow
  # ) +
  #   geom_text(aes(y = mean - 0.2, label = wflow_id), angle = 90, hjust = 1) +
  #   lims(y = c(0.6, 1.4)) +
  #   theme(legend.position = "none")
  
  set.seed(seed)
  matched_results <- 
    rank_results(race_results, select_best = TRUE) |>  
    select(wflow_id, .metric, race = mean, config_race = .config) |> 
    inner_join(
      rank_results(grid_results, select_best = TRUE) |>  
        select(wflow_id, .metric, complete = mean, 
               config_complete = .config, model),
      by = c("wflow_id", ".metric"),
    ) |>   
    filter(.metric == "rmse")
  
  # matched_results
  # 
  # matched_results |>  
  #   ggplot(aes(x = complete, y = race)) + 
  #   geom_abline(lty = 3) + 
  #   geom_point() + 
  #   geom_text_repel(aes(label = model)) +
  #   coord_obs_pred() + 
  #   labs(x = "Complete Grid RMSE", y = "Racing RMSE") 
  
  # Create a list to store predictions for all 12 models
  all_model_predictions <- list()
  
  # Loop through the 12 models from matched_results and collect predictions
  for (i in seq_len(nrow(matched_results))) {
    
    # Extract the model information
    model_id <- matched_results$wflow_id[i]
    
    set.seed(seed)
    # Get the best configuration for the current model
    best_results <- race_results |>  
      extract_workflow_set_result(model_id) |> 
      select_best(metric = "rmse")
    
    # Finalize the model with the best hyperparameters
    set.seed(seed)
    final_model <- race_results |> 
      extract_workflow(model_id) |>  
      finalize_workflow(best_results) |>  
      fit(data = calibration_data)
    
    # Generate predictions for calibration and verification datasets
    calibration_preds <- predict(final_model, calibration_data) |> bind_cols(calibration_data)
    verification_preds <- predict(final_model, verification_data) |> bind_cols(verification_data)
    
    # Add a column to distinguish datasets and model
    calibration_preds <- calibration_preds |> mutate(model = model_id, dataset = "calibration")
    verification_preds <- verification_preds |> mutate(model = model_id, dataset = "verification")
    
    # Combine predictions into a single dataframe
    all_model_predictions[[i]] <- bind_rows(calibration_preds, verification_preds)
  }
  
  # Combine all predictions into a single dataframe
  all_model_predictions_df <- bind_rows(all_model_predictions) |> 
    relocate(Country, .before = everything())  # Move 'Country' to the first column
  
  # Create a ggplot2 plot with facet wrapping for all models
  # ggplot(all_model_predictions_df, aes(x = log_BFA1000, y = .pred, color = dataset)) +
  #   geom_point(alpha = 0.6) +
  #   geom_abline(slope = 1, intercept = 0, color = "gray50", lty = 2) +  # Ideal line
  #   facet_wrap(~model, ncol = 4, nrow = 3) +  # Facet wrap for all models (4x3 grid)
  #   labs(title = "Predicted vs Observed Values for 12 Models",
  #        x = "Observed log_BFA1000",
  #        y = "Predicted log_BFA1000") +
  #   theme_bw() +
  #   scale_color_manual(values = c("calibration" = "blue", "verification" = "red"))
  
  # --------------
  # Ensemble mean
  if(metamodel == FALSE){
    # set.seed(seed)
    # models_stack <-
    #   stacks() |>
    #   add_candidates(race_results)
    # 
    # models_stack
    
    set.seed(seed)
    
    if (use_perturbation_audit) {
      
      scenario_audit_data <- prepare_scenario_audit_data(
        scenario_root = "./inputs",
        countries = scenario_settings$countries,
        clean_data = clean_data,
        scenario = "ssp585"
      )
      
      if (isTRUE(perturbation_audit_settings$use_scenario_audit)) {
        
        perturbation_audit_settings$max_allowed_scenario_BFAI <-
          quantile(exp(clean_data$log_BFA1000), 0.99, na.rm = TRUE) *
          perturbation_audit_settings$scenario_bfai_multiplier
      }
      
      perturbation_audit_output <- run_perturbation_audit_for_stacking(
        race_results = race_results,
        matched_results = matched_results,
        calibration_data = calibration_data,
        predictors = predictors,
        out_path = out_path,
        settings = perturbation_audit_settings,
        scenario_audit_data = scenario_audit_data,
        seed = seed
      )
      
      # First try: use the prescribed perturbation-audit settings
      # perturbation_audit_output <- run_perturbation_audit_for_stacking(
      #   race_results = race_results,
      #   matched_results = matched_results,
      #   calibration_data = calibration_data,
      #   predictors = predictors,
      #   out_path = out_path,
      #   settings = perturbation_audit_settings,
      #   seed = seed
      # )
      
      # n_stack_candidates <- perturbation_audit_output$race_results_for_stack |>
      #   stacks::stacks() # dummy-safe placeholder; do not use
      
      n_passed_candidates <- perturbation_audit_output$robustness_table |>
        dplyr::filter(status == "pass") |>
        nrow()
      
      # If too few candidates passed, relax the audit once.
      # This avoids blend_predictions() failing with only one candidate.
      if (n_passed_candidates < 3) {
        
        log_message(
          "Only ", n_passed_candidates,
          " candidate(s) passed the perturbation audit. ",
          "Repeating audit with relaxed settings."
        )
        
        perturbation_audit_settings_relaxed <- modifyList(
          perturbation_audit_settings,
          list(
            max_multiplier = perturbation_audit_settings$max_multiplier * 0.8,
            max_log_increase = perturbation_audit_settings$max_log_increase * 1.5,
            max_step_jump = perturbation_audit_settings$max_step_jump * 1.5,
            min_monotonic_share = perturbation_audit_settings$min_monotonic_share * 0.9
          )
        )
        
        perturbation_audit_output <- run_perturbation_audit_for_stacking(
          race_results = race_results,
          matched_results = matched_results,
          calibration_data = calibration_data,
          predictors = predictors,
          out_path = out_path,
          settings = perturbation_audit_settings_relaxed,
          seed = seed
        )
      }
      
      race_results_for_stack <- perturbation_audit_output$race_results_for_stack
      robustness_table       <- perturbation_audit_output$robustness_table
      robust_model_ids       <- perturbation_audit_output$robust_model_ids
      matched_results_for_scenarios <- perturbation_audit_output$matched_results_for_scenarios
      
    } else {
      
      race_results_for_stack <- race_results
      robustness_table <- NULL
      robust_model_ids <- matched_results$wflow_id
      matched_results_for_scenarios <- matched_results
    }
    
    models_stack <- stacks() |>
      add_candidates(race_results_for_stack)
    
    # filtered_results <- race_results |>
    #   filter(wflow_id != "MARS")
    #
    # models_stack <- stacks() |>
    #   add_candidates(filtered_results)
    
    # Calculate RMSE for each model column
    rmse_results <- models_stack |>
      summarise(across(-log_BFA1000,
                       ~ yardstick::rmse_vec(models_stack$log_BFA1000, .x),
                       .names = "RMSE_{.col}")) |>
      pivot_longer(cols = everything(),
                   names_to = "Model",
                   values_to = "RMSE") |>
      arrange(RMSE)
    
    # Display RMSE results sorted
    print(rmse_results)
    
    # View(rmse_results)
    
    set.seed(seed+1)
    ens <- blend_predictions(models_stack)
    
    # autoplot(ens)
    
    set.seed(seed+2)
    ens <- blend_predictions(models_stack, penalty = 10^seq(-2, -0.5, length = 20))
    
    #---------------------------------------------------------------------------
    
    # n_ens_reps – Repeat ensemble blending multiple times with different seeds
    # to improve stability and select the best-performing stack
    
    all_ens_models <- list()
    rmse_scores <- numeric(n_ens_reps)
    
    reg_metrics <- metric_set(rmse)
    
    for (i in 1:n_ens_reps) {
      set.seed(seed + i)
      
      ens_tmp <- blend_predictions(models_stack, penalty = 10^seq(-2, -0.5, length = 20))
      
      # Fit members before evaluation
      ens_tmp <- fit_members(ens_tmp)
      
      # Predict on calibration data and compute RMSE
      pred_tmp <- predict(ens_tmp, calibration_data) |>
        bind_cols(calibration_data)
      
      rmse_val <- reg_metrics(pred_tmp, truth = log_BFA1000, estimate = .pred) |>
        filter(.metric == "rmse") |>
        pull(.estimate)
      
      rmse_scores[i] <- rmse_val
      all_ens_models[[i]] <- ens_tmp
    }
    
    # Select the best ensemble (lowest RMSE on calibration set)
    best_index <- which.min(rmse_scores)
    ens <- all_ens_models[[best_index]]
    log_message("Selected ensemble model from seed index ", best_index,
            " with RMSE: ", round(rmse_scores[best_index], 4))
    
    #---------------------------------------------------------------------------
    
    # autoplot(ens)
    
    stack_rank <- autoplot(ens, "weights") +
      geom_text(aes(x = weight + 0.01, label = model), hjust = 0) +
      theme(legend.position = "none") +
      lims(x = c(-0.01, 0.8))
    
    # Step 1: Extract weights plot data
    weights_data <- autoplot(ens, "weights")$data
    
    # Step 2: Use ens$cols_map to reverse-map model names
    # Create lookup: every member ID maps to readable name
    lookup <- purrr::imap_dfr(ens$cols_map, ~ tibble(terms = .x, label = .y))
    
    # Step 3: Join and collapse to readable names
    weights_labeled <- weights_data |>
      left_join(lookup, by = "terms") |>
      mutate(label = factor(label, levels = unique(label)))  # optional
    
    weights_labeled <- weights_labeled |>
      dplyr::filter(is.finite(weight), weight > 1e-6) |>
      dplyr::arrange(desc(weight)) |>
      dplyr::mutate(member_rank = dplyr::row_number())  # this will be used as y-axis
    
    stack_rank <- ggplot(
      weights_labeled,
      aes(x = weight, y = reorder(member_rank, weight))
    ) +
      geom_col(aes(fill = label)) +
      geom_text(aes(x = weight + 0.01, label = label), hjust = 0) +
      theme_bw() +
      theme(legend.position = "none") +
      labs(
        title = paste0("Penalty = ", signif(ens$penalty$penalty, 3)),
        x = "Stacking Coefficient",
        y = "Member"
      ) +
      coord_cartesian(xlim = c(0, max(weights_labeled$weight, na.rm = TRUE) * 2.5))
    
    # Save the plot
    if(use_country == TRUE){
      plot_name <- paste0(out_path, "/stack_rank_ensemble_country.png")
    }else{
      plot_name <- paste0(out_path, "/stack_rank_ensemble.png")
    }
    ggsave(plot_name, plot = stack_rank, width = 1 * 140, height = 1 * 130, dpi = 600, units = 'mm', device = ragg::agg_png)
    
    set.seed(seed)
    ens <- fit_members(ens)
    
    reg_metrics <- metric_set(rmse, rsq)
    
    ens_calibration_pred <-
      predict(ens, calibration_data) |>
      bind_cols(calibration_data) |>
      mutate(dataset = "calibration")
    
    ens_calibration_pred |>
      reg_metrics(log_BFA1000, .pred)
    
    ens_verification_pred <-
      predict(ens, verification_data) |>
      bind_cols(verification_data) |>
      mutate(dataset = "verification")
    
    ens_verification_pred |>
      reg_metrics(log_BFA1000, .pred)
    
    # Combine ens predictions into a single dataframe
    ens_model_pred_df <- bind_rows(ens_calibration_pred, ens_verification_pred)
    
  }else{
    # Apply metamodel ensemble
    metafit <- metafit_ens(race_results, calibration_data, verification_data, rmse_deviation = 1.1, model = metamodel)
    
    ens_model_pred_df <- metafit$predictions
    ens <- metafit$ensemble
  }
  
  # Calculate RMSE and R-squared for calibration and verification datasets using yardstick
  metrics_values <- ens_model_pred_df |>
    group_by(dataset) |>
    summarize(
      rmse = rmse_vec(truth = log_BFA1000, estimate = .pred),
      r2 = rsq_vec(truth = log_BFA1000, estimate = .pred)
    ) |>
    mutate(metric_label = case_when(
      dataset == "calibration"  ~ paste0("Calibration: RMSE == ", round(rmse, 2),
                                         " ~ \";\" ~ R^2 == ", round(r2, 2)),
      dataset == "verification" ~ paste0("Verification: RMSE == ", round(rmse, 2),
                                         " ~ \";\" ~ R^2 == ", round(r2, 2))
    ))
  
  # Adjust positions for the metrics labels
  metrics_values <- metrics_values |>
    mutate(y_position = ifelse(dataset == "calibration",
                               max(ens_model_pred_df$.pred) - 0.1,
                               max(ens_model_pred_df$.pred) - 0.5))
  
  # Create the ggplot with RMSE and R-squared annotations
  # ggplot(ens_model_pred_df, aes(x = log_BFA1000, y = .pred, color = dataset)) +
  #   geom_point(alpha = 0.6) +
  #   geom_abline(slope = 1, intercept = 0, color = "gray50", lty = 2) +  # Ideal line
  #   labs(title = "Predicted vs Observed Values - Model Ensemble Mean",
  #        x = "Observed log_BFA1000",
  #        y = "Predicted log_BFA1000") +
  #   theme_bw() +
  #   scale_color_manual(values = c("calibration" = "blue", "verification" = "red")) +
  #   geom_text(data = metrics_values, aes(x = min(ens_model_pred_df$log_BFA1000), 
  #                                        y = y_position,
  #                                        label = metric_label),
  #             hjust = 0, color = "black", size = 4, parse = TRUE)
  
  # Prepare RMSE values with multiline text using ggtext styling
  rmse_values_combined <- metrics_values |> 
    select(dataset, rmse) |> 
    mutate(combined_label = paste0(
      "RMSE<sub>calibration</sub> = ", sprintf("%.2f", rmse[dataset == "calibration"]), "<br>",
      "RMSE<sub>verification</sub> = ", sprintf("%.2f", rmse[dataset == "verification"])
    ),
    y_position = 5.5
    )
  
  # Set country levels
  country_levels <- unique(ens_model_pred_df$Country)
  custom_colors_country <- custom_colors_country[country_levels]
  
  # Create the main plot with the legend
  num_countries <- length(unique(ens_model_pred_df$Country))
  main_plot_ens_pred <- ggplot(ens_model_pred_df, aes(x = log_BFA1000, y = .pred, color = Country, shape = Country)) +
    geom_point(aes(fill = ifelse(dataset == "verification", Country, NA)), 
               size = 3, alpha = 0.8, stroke = 1) +  # stroke adjusts border width
    scale_shape_manual(
      values = rep(c(21, 22, 24), length.out = length(unique(ens_model_pred_df$Country))),
      name = "Country"
    ) +
    scale_color_manual(
      values = custom_colors_country,  # Apply custom colors for outline
      name = "Country"
    ) +
    scale_fill_manual(
      values = custom_colors_country,  # Apply the same custom colors for fill (only when `Verification`)
      na.translate = FALSE,    # Avoid showing NA in the legend
      guide = "none"           # Suppress legend for fill
    ) +
    geom_abline(intercept = 0, slope = 1, color = "#2b2b2b", linetype = "dashed", size = 1) +
    # scale_x_log10() +  # Set x-axis to logarithmic scale
    # scale_y_log10() +  # Set y-axis to logarithmic scale
    scale_x_continuous(
      limits = c(-6.5, 6.5),  # Set x-axis limits
      expand = c(0, 0)        # Remove buffer
    ) +
    scale_y_continuous(
      limits = c(-6.5, 6.5),  # Set y-axis limits
      expand = c(0, 0)        # Remove buffer
    ) +
    # Using geom_richtext from ggtext for proper subscripts and multiline rendering
    geom_richtext(
      data = rmse_values_combined, 
      aes(label = combined_label, x = -6, y = y_position),  
      inherit.aes = FALSE,
      fill = "white",  
      color = "#2b2b2b", 
      size = 4,
      hjust = 0,  # Forces entire box alignment to the left
      label.size = 0.5,
      # CSS for internal left alignment of the text
      label.padding = unit(0.2, "lines")
    ) +
    labs(
      title = "Observed vs. Predicted Values",
      x = "Observed log(BFAI)",
      y = "Predicted log(BFAI)",
      color = "Country",
      shape = "Country"
    ) +
    guides(
      color = guide_legend(
        override.aes = list(
          fill = custom_colors_country
        )
      )
    ) +
    theme_bw() +
    theme(
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 10)
    )
  
  # Extract the Country legend from the main_plot_ens_pred
  country_legend <- get_legend(main_plot_ens_pred)
  
  # Create dummy data for Dataset legend (empty vs. filled)
  dataset_legend_data <- data.frame(
    x = c(1, 1),
    y = c(1, 2),
    Dataset = factor(c("calibration", "verification"))
  )
  
  # Create dummy plot for Dataset legend
  dataset_legend_plot <- ggplot(dataset_legend_data, aes(x = x, y = y, fill = Dataset, shape = Dataset)) +
    geom_point(size = 3, stroke = 1, color = "#2b2b2b") +
    scale_fill_manual(
      values = c("calibration" = "white", "verification" = "#2b2b2b"),
      name = "Dataset",
      labels = c("Calibration", "Verification")
    ) +
    scale_shape_manual(
      values = c(21, 21),
      name = "Dataset",
      labels = c("Calibration", "Verification")
    ) +
    theme_bw() +
    theme(
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 10)
    )
  
  # Extract the Dataset legend
  dataset_legend <- get_legend(dataset_legend_plot)
  
  # Remove legends from the main plot
  main_plot_ens_pred <- main_plot_ens_pred +
    theme(legend.position = "none")
  
  # Standardize widths
  max_width <- grid::unit.pmax(country_legend$widths, dataset_legend$widths)
  country_legend$widths <- max_width
  dataset_legend$widths <- max_width
  
  # create a blank plot for legend alignment 
  spacer <- plot_spacer() + theme_void()
  
  # Combine both legends
  legends <- plot_grid(
    spacer,
    grid::grobTree(country_legend),
    spacer,
    grid::grobTree(dataset_legend),
    spacer,
    ncol = 1,
    nrow = 5,
    align = "hv",
    axis = "l",
    rel_heights = c(0.13, 0.4, 0.15, 0.3, 0.02)
  )
  
  # Final plot
  final_plot_ens_pred <- plot_grid(
    main_plot_ens_pred,
    legends,
    nrow = 1,
    align = "h",
    axis = "t",
    rel_widths = c(0.65, 0.2)
  ) + theme_bw() +
    theme(panel.border = element_blank())
  
  # Save the plot
  if(use_country == TRUE){
    plot_name <- paste0(out_path, "/ensemble_models_predictions_log-scale_country.png")
  }else{
    plot_name <- paste0(out_path,"/ensemble_models_predictions_log-scale.png")
  }
  ggsave(plot_name, plot = final_plot_ens_pred, width = 1 * 160, height = 1 * 130, dpi = 600, units = 'mm', device = ragg::agg_png)
  
  #-------------------------------------------------------------------------------
  
  # Compute RMSE for each model and dataset
  rmse_values <- all_model_predictions_df |> 
    group_by(model, dataset) |> 
    summarise(rmse = rmse_vec(truth = log_BFA1000, estimate = .pred), .groups = "drop")
  
  # Prepare RMSE values with multiline text using ggtext styling
  rmse_values_combined <- rmse_values |> 
    group_by(model) |> 
    summarize(
      combined_label = paste0(
        "RMSE<sub>calibration</sub> = ", sprintf("%.2f", rmse[dataset == "calibration"]), "<br>",
        "RMSE<sub>verification</sub> = ", sprintf("%.2f", rmse[dataset == "verification"])
      ),
      y_position = 5.1
    )
  
  range(all_model_predictions_df$log_BFA1000)
  range(all_model_predictions_df$.pred)
  
  # Create the main plot with the legend
  num_countries <- length(unique(all_model_predictions_df$Country))
  main_plot_with_legend <- ggplot(all_model_predictions_df, aes(x = log_BFA1000, y = .pred, color = Country, shape = Country)) +
    geom_point(aes(fill = ifelse(dataset == "verification", Country, NA)), 
               size = 3, alpha = 0.8, stroke = 1) +  # stroke adjusts border width
    facet_wrap(~ model, nrow = 4, ncol = 3) +  # Create facets for each model
    scale_shape_manual(
      values = rep(c(21, 22, 24), length.out = length(unique(all_model_predictions_df$Country))),
      name = "Country"
    ) +
    scale_color_manual(
      values = custom_colors_country,  # Apply custom colors for outline
      name = "Country"
    ) +
    scale_fill_manual(
      values = custom_colors_country,  # Apply the same custom colors for fill (only when `Verification`)
      na.translate = FALSE,    # Avoid showing NA in the legend
      guide = "none"           # Suppress legend for fill
    ) +
    geom_abline(intercept = 0, slope = 1, color = "#2b2b2b", linetype = "dashed", size = 1) +
    # scale_x_log10() +  # Set x-axis to logarithmic scale
    # scale_y_log10() +  # Set y-axis to logarithmic scale
    scale_x_continuous(
      limits = c(-6.5, 6.5),  # Set x-axis limits
      expand = c(0, 0)        # Remove buffer
    ) +
    scale_y_continuous(
      limits = c(-6.5, 6.5),  # Set y-axis limits
      expand = c(0, 0)        # Remove buffer
    ) +
    # Using geom_richtext from ggtext for proper subscripts and multiline rendering
    geom_richtext(
      data = rmse_values_combined, 
      aes(label = combined_label, x = -6, y = y_position),  
      inherit.aes = FALSE,
      fill = "white",  
      color = "#2b2b2b", 
      size = 4,
      hjust = 0,  # Forces entire box alignment to the left
      label.size = 0.5,
      # CSS for internal left alignment of the text
      label.padding = unit(0.2, "lines")
    ) +
    labs(
      title = "Observed vs. Predicted Values",
      x = "Observed log(BFAI)",
      y = "Predicted log(BFAI)",
      color = "Country",
      shape = "Country"
    ) +
    guides(
      color = guide_legend(
        ncol = 1,
        override.aes = list(fill = custom_colors_country)
      ),
      shape = guide_legend(ncol = 1)
    )+
    theme_bw() +
    theme(
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 10),
      strip.text = element_text(size = 10, face = "bold")
    )
  
  # Extract the Country legend
  country_legend <- get_legend(main_plot_with_legend)
  
  # Create a dummy dataset for the Dataset legend
  dataset_legend_data <- data.frame(
    x = c(1, 1),  # Dummy x-axis
    y = c(1, 2),  # Dummy y-axis
    Dataset = factor(c("calibration", "verification"))  # Treat Dataset as a factor
  )
  
  # Create a dummy plot for the Dataset legend
  dataset_legend_plot <- ggplot(dataset_legend_data, aes(x = x, y = y, fill = Dataset, shape = Dataset)) +
    geom_point(size = 3, stroke = 1, color = "#2b2b2b") +
    scale_fill_manual(
      values = c("calibration" = "white", "verification" = "#2b2b2b"),  # Empty for Calibration, black for Verification
      name = "Dataset",
      labels = c("Calibration", "Verification")
    ) +
    scale_shape_manual(
      values = c(21, 21),  # Circle shapes for both
      name = "Dataset",
      labels = c("Calibration", "Verification")
    ) +
    theme_bw() +
    theme(
      legend.position = "right",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 10)
    )
  
  # Extract the Dataset legend
  dataset_legend <- get_legend(dataset_legend_plot)
  
  # Remove legends from the main plot
  main_plot <- main_plot_with_legend +
    theme(legend.position = "none")
  
  # Standardize widths of the legend gtables
  max_width <- grid::unit.pmax(country_legend$widths, dataset_legend$widths)
  country_legend$widths <- max_width
  dataset_legend$widths <- max_width
  
  # create a blank plot for legend alignment 
  spacer <- plot_spacer() + theme_void()
  
  legends <- plot_grid(
    spacer,                             # Spacer at the top
    grid::grobTree(country_legend),     # First legend
    spacer,                             # Spacer in between
    grid::grobTree(dataset_legend),     # Second legend
    spacer,                             # Spacer at the bottom
    ncol = 1,                           # One column layout
    nrow = 5,                           # Four rows for flexibility
    align = "hv",                       # Vertically align the legends
    axis = "l",                         # Align elements along the left axis
    rel_heights = c(4, 3, 0.4, 1, 0.01)  # Adjust these to control spacing/alignment
  )
  
  # Combine the two legends into one
  final_p <- plot_grid(main_plot,
                       legends,
                       nrow = 1,
                       align = "h",
                       axis = "t",
                       rel_widths = c(1, 0.15)
  ) + theme_bw() +
    theme(
      panel.border = element_blank()     # Remove the black border around the plot
    )
  
  # print(final_p)
  
  # Save the plot
  if(use_country == TRUE){
    plot_name <- paste0(out_path, "/all_models_predictions_log-scale_country.png")
  }else{
    plot_name <- paste0(out_path, "/all_models_predictions_log-scale.png")
  }
  ggsave(plot_name, plot = final_p, width = 3 * 125*0.8, height = 4 * 110*0.8, dpi = 600, units = 'mm', device = ragg::agg_png)
  
  #-------------------------------------------------------------------------------
  # Climate scenario
  
  #------------------------
  # Fitting scenario models
  log_message("Fitting scenario models")
  
  scenario_models <- list()
  
  for (i in seq_len(nrow(matched_results_for_scenarios))) {
  
    model_id <- matched_results_for_scenarios$wflow_id[i]
    
    log_message("Fitting scenario model: ", model_id)
    
    best_results <- race_results_for_stack |>
      extract_workflow_set_result(model_id) |>
      select_best(metric = "rmse")
    
    scenario_models[[model_id]] <- race_results_for_stack |>
      extract_workflow(model_id) |>
      finalize_workflow(best_results) |>
      fit(data = calibration_data)
    
    rm(best_results)
    gc()
  }
  
  #--------------------------------------
  # Precomputing all scenario predictions
  log_message("Precomputing all scenario predictions")
  
  all_scenario_predictions <- list()
  all_ensemble_predictions <- list()
  
  for (country_code in scenario_settings$countries) {
    
    log_message("Precomputing scenario predictions for: ", country_code)
    
    scenario_root <- "./inputs"
    
    scenario_country_files <- list.files(
      path = scenario_root,
      pattern = paste0(country_code, "_forest\\.csv$"),
      recursive = TRUE,
      full.names = TRUE
    ) |>
      keep(\(x) str_detect(x, "scenariomip"))
    
    scenario_country <- scenario_country_files |>
      map_dfr(\(file) {
        
        path_parts <- str_split(file, "/", simplify = TRUE)
        
        read_csv(file, show_col_types = FALSE) |>
          pivot_longer(
            cols = matches("^\\d{4}$"),
            names_to = "Year",
            values_to = "value"
          ) |>
          mutate(
            Country = country_code,
            Dataset = "scenario",
            Scenario = str_extract(file, "ssp\\d+"),
            GCM = path_parts[length(path_parts) - 2],
            Period = as.numeric(path_parts[length(path_parts) - 1]),
            Year = as.numeric(Year),
            varname = paste(Variable, Season, sep = "_")
          ) |>
          select(Country, Scenario, GCM, Period, Year, Dataset, varname, value)
      }) |>
      pivot_wider(
        names_from = varname,
        values_from = value
      ) |>
      arrange(Scenario, GCM, Period, Year)
    
    country_summary <- clean_data |>
      filter(Country == country_code) |>
      summarise(
        Pines = mean(Pines, na.rm = TRUE),
        Conifers = mean(Conifers, na.rm = TRUE),
        Broadleaved = mean(Broadleaved, na.rm = TRUE)
      )
    
    scenario_country_model <- scenario_country |>
      mutate(
        Country = country_code,
        Pines = country_summary$Pines,
        Conifers = country_summary$Conifers,
        Broadleaved = country_summary$Broadleaved
      ) |>
      rename_with(
        \(x) x |>
          str_replace_all("\\+", "_plus") |>
          str_replace_all("-", "_")
      )
    
    for (model_id in names(scenario_models)) {
      
      tmp <- stats::predict(
        scenario_models[[model_id]],
        new_data = scenario_country_model
      ) |>
        bind_cols(scenario_country_model) |>
        mutate(
          ML_model = model_id,
          predicted_BFA1000 = exp(.pred)
        )
      
      all_scenario_predictions[[paste(country_code, model_id, sep = "_")]] <- tmp
    }
    
    tmp_ens <- stats::predict(
      ens,
      new_data = scenario_country_model
    ) |>
      bind_cols(scenario_country_model) |>
      mutate(
        ML_model = "ensemble mean",
        predicted_BFA1000 = exp(.pred)
      )
    
    all_ensemble_predictions[[country_code]] <- tmp_ens
  }
  
  all_scenario_predictions_df <- bind_rows(all_scenario_predictions)
  all_ensemble_predictions_df <- bind_rows(all_ensemble_predictions)
  
  plot_data_path <- file.path(out_path, "plot_data")
  dir.create(plot_data_path, recursive = TRUE, showWarnings = FALSE)
  
  for (country_code in scenario_settings$countries) {
    saveRDS(
      all_scenario_predictions_df |> filter(Country == country_code),
      file.path(plot_data_path, paste0(country_code, "_scenario_predictions.rds"))
    )
    
    saveRDS(
      all_ensemble_predictions_df |> filter(Country == country_code),
      file.path(plot_data_path, paste0(country_code, "_ensemble_predictions.rds"))
    )
  }
  
  saveRDS(all_scenario_predictions_df, file.path(out_path, "all_scenario_predictions_df.rds"))
  saveRDS(all_ensemble_predictions_df, file.path(out_path, "all_ensemble_predictions_df.rds"))
  saveRDS(clean_data, file.path(out_path, "clean_data_for_plots.rds"))
  saveRDS(calibration_data, file.path(out_path, "calibration_data_for_plots.rds"))
  saveRDS(scenario_settings, file.path(out_path, "scenario_settings.rds"))
  saveRDS(use_country, file.path(out_path, "use_country.rds"))
  saveRDS(num_cores_plot, file.path(out_path, "num_cores_plot.rds"))
  
  rm(all_scenario_predictions, all_ensemble_predictions, scenario_models)
  gc()
  
  #-----------------------------------------------------------------------------
  # --- RMSE ---
  rmse_individual <- all_model_predictions_df |> 
    group_by(model, dataset) |> 
    summarise(rmse = rmse_vec(truth = log_BFA1000, estimate = .pred), .groups = "drop")
  
  rmse_ensemble <- ens_model_pred_df |> 
    group_by(dataset) |> 
    summarise(rmse = rmse_vec(truth = log_BFA1000, estimate = .pred), .groups = "drop") |> 
    mutate(model = "ensemble")
  
  rmse_all <- bind_rows(rmse_individual, rmse_ensemble) |> 
    pivot_wider(names_from = dataset, values_from = rmse, names_prefix = "rmse_")
  
  # --- R-squared ---
  rsq_individual <- all_model_predictions_df |> 
    group_by(model, dataset) |> 
    summarise(rsq = rsq_vec(truth = log_BFA1000, estimate = .pred), .groups = "drop")
  
  rsq_ensemble <- ens_model_pred_df |> 
    group_by(dataset) |> 
    summarise(rsq = rsq_vec(truth = log_BFA1000, estimate = .pred), .groups = "drop") |> 
    mutate(model = "ensemble")
  
  rsq_all <- bind_rows(rsq_individual, rsq_ensemble) |> 
    pivot_wider(names_from = dataset, values_from = rsq, names_prefix = "rsq_")
  
  # --- Combine RMSE and R² into one table ---
  error_stat <- rmse_all |> 
    left_join(rsq_all, by = "model") |> 
    select(model, rmse_calibration, rmse_verification, rsq_calibration, rsq_verification)
  
  # --- Save to CSV ---
  write_csv(error_stat, paste0(out_path, "/error_stat.csv"))
  
  # Save RMSE summary table as a CSV file for external review or reporting
  write_csv(error_stat, paste0(out_path, "/error_stat.csv"))
  
  # Save full tuning results from grid search (slower but exhaustive)
  saveRDS(grid_results, file = paste0(out_path, "/grid_results.rds"))
  
  # Save full tuning results from racing (faster, early-stopping based tuning)
  saveRDS(race_results, file = paste0(out_path, "/race_results.rds"))
  
  # Save the comparison between the best configurations from racing and grid search
  saveRDS(matched_results, file = paste0(out_path, "/matched_results.rds"))
  
  # Save the full model stack (only if metamodel ensemble is not used)
  if(metamodel == FALSE){
    saveRDS(models_stack, file = paste0(out_path, "/models_stack.rds"))
  }
  
  # Save the final ensemble model (either blended or meta-model-based)
  saveRDS(ens, file = paste0(out_path, "/ensemble_model.rds"))
  
  # Remove the objects to free environment before furrr parallel plotting
  rm(list = intersect(
    c(
      "ens", "models_stack", "race_results", "grid_results",
      "matched_results", "race_results_for_stack", "scenario_models",
      "rmse_individual", "rmse_ensemble", "rmse_all",
      "rsq_individual", "rsq_ensemble", "rsq_all", "error_stat"
    ),
    ls()
  ))
  gc()
  
  #------------------
  # Parallel plotting
  callr::rscript(
    script = "plot_scenarios_worker.R",
    cmdargs = out_path,
    show = TRUE
  )
  
  log_message(basename(out_path), " completed successfully.")
}