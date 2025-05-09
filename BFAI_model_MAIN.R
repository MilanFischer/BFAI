################################################################################
# Must read
# https://github.com/stevenpawley/colino
# https://www.tmwr.org/
# https://www.tidymodels.org/find/recipes/
################################################################################

# Ideas
# If use_country == TRUE -> exclude broadleaved, coniferous and pines – done
# Show correlation matrix of predictors for 001 and 033
# Add mean line for future prediction figure
# Ask for annual data – done
# Can we get additional countries – in the future  yes, we will see how long it will take.

# for(run_ID in 3:15){
for(run_ID in 1:33){
  # Clear workspace
  rm(list = setdiff(ls(), "run_ID"))
  
  # Load necessary libraries for data manipulation, modeling, and visualization
  source("./src/load_libraries.R")
  source("./src/colors.R")
  source("./src/model_specification.R")
  source("./src/corellation_filter.R")
  source("./src/boruta.R")
  source("./src/metafit_ens.R")
  source("./src/files_manage.R")
  
  # Define the target variable
  target <- "log_BFA1000"
  
  # Choose whether to use country as the predictor
  use_country <- TRUE
  
  # Apply metamodel ensemble mean: FALSE or "lm", "glm", "xgboost", "nnet", "glmnet", "ranger", "cubist", "bart"
  metamodel <- FALSE
  
  # Threshold for removing highly correlated predictors
  cor_thresh = 0.9
  
  # Set and create output path
  # out_path <- create_next_output_dir()
  out_path <- paste0("./outputs/out_", sprintf("%03d", run_ID))
  
  # Separate calibration and verification datasets
  calibration_years <- seq(1990, 2022)[(seq(1990, 2022) - 1990) %% 3 != 2]  # 1st and 2nd years in each 3-year block
  verification_years <- seq(1990, 2022)[(seq(1990, 2022) - 1990) %% 3 == 2]  # 3rd year in each 3-year block
  
  # Set up parallel processing (to speed up computations)
  num_cores <- parallel::detectCores(logical = TRUE) - 1  # Use all but 1 core
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  # Load data and handle missing values
  # Ensure proper handling of missing values
  data <- read_excel("./inputs/BFAI_inputs.xlsx", 
                     sheet = "inputs", 
                     na = c("", "NA", "N/A", "#N/A"))
  
  # Replace infinite values with NA and drop them
  clean_data <- data |> 
    mutate(across(where(is.numeric), ~na_if(., Inf))) |>  # Replace Inf with NA
    drop_na()
  
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
    
    target <- config$target
    
    cor_thresh <- config$cor_thresh
    
    metamodel <- config$metamodel
    
    seed <- config$seed
    
    message("Loaded config from: ", config_path)
  } else {
    
    message("No config.yml found. Generating new config.")
    
    seed <- 1234
    
    # Remove highly correlated predictors 
    cor_filtered_cols <- corr_filter(calibration_data, cor_thresh)
    
    boruta_filtered_cols <- get_boruta_predictors(calibration_data |> select(all_of(cor_filtered_cols)),
                                                  target = target, use_country = use_country, min_freq = 5, v = 10, seed = 1234)
    
    predictors_only <- setdiff(boruta_filtered_cols, c(target, "Year"))
    
    # RFE with correct predictors
    ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
    rfe_result <- rfe(calibration_data[, predictors_only], calibration_data[[target]],
                      sizes = length(predictors_only), rfeControl = ctrl)
    
    # Extract variable importance
    vip_df <- data.frame(Predictors = row.names(varImp(rfe_result)),
                         Importance = varImp(rfe_result)$Overall)
    
    # Sort by importance
    vip_df <- vip_df[order(-vip_df$Importance), ]
    
    # Plot
    ggplot(vip_df, aes(x = reorder(Predictors, Importance), y = Importance)) +
      geom_col() +
      coord_flip() +
      labs(title = "Variable Importance (RFE - Random Forest)",
           x = "Variable", y = "Importance") +
      theme_minimal()
    
    final_cols <- c("Year", target, vip_df$Predictors[1:run_ID])
    
    calibration_data <- calibration_data |> select(all_of(final_cols))
    verification_data <- verification_data |> select(all_of(final_cols))
    
    predictors <- setdiff(final_cols, c("Year", target))
    
    # Save predictors and seed to config.yml
    config <- list(
      predictors = predictors,
      use_country =use_country,
      target = target,
      cor_thresh = cor_thresh,
      metamodel = metamodel,
      seed = seed
    )
    
    write_yaml(config, config_path)
    message("Saved new config to: ", config_path)
  }
  
  # Create a formula dynamically
  formula <- as.formula(paste(target, "~", paste(predictors, collapse = " + ")))
  
  if (use_country == TRUE) {
    no_pre_proc_rec <- recipe(formula, data = calibration_data) |>
      step_nzv(all_predictors()) |>
      step_lencode_glm(Country, outcome = target) 
  } else {
    no_pre_proc_rec <- recipe(formula, data = calibration_data) |>
      # Retain Country even if not used
      update_role(Country, new_role = "ID") |> 
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
      parallel_over = "everything",
      save_workflow = TRUE
    )
  
  # Run the grid search
  grid_results <-
    all_workflows |>
    workflow_map(
      seed = seed,
      resamples = cv_folds,
      grid = 50, # Reduced grid size for debugging, otherwise set >50
      control = grid_ctrl
    )
  
  # Verify results and notes again
  show_notes(grid_results)
  grid_results |>  
    rank_results() |>  
    filter(.metric == "rmse") |>  
    select(model, .config, rmse = mean, rank) |> View()
  
  autoplot(
    grid_results,
    rank_metric = "rmse",  # <- how to order models
    metric = "rmse",       # <- which metric to visualize
    select_best = TRUE     # <- one point per workflow
  ) +
    geom_text(aes(y = mean - 0.2, label = wflow_id), angle = 90, hjust = 1) +
    lims(y = c(0.6, 1.4)) +
    theme(legend.position = "none")
  
  race_ctrl <-
    control_race(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = TRUE
    )
  
  race_results <-
    all_workflows |>
    workflow_map(
      "tune_race_anova",
      seed = 123,
      resamples = cv_folds,
      grid = 100,
      control = race_ctrl
    )
  
  race_results |>  
    rank_results() |>  
    filter(.metric == "rmse") |>  
    select(model, .config, rmse = mean, rank) |> View()
  
  autoplot(
    race_results,
    rank_metric = "rmse",  # <- how to order models
    metric = "rmse",       # <- which metric to visualize
    select_best = TRUE     # <- one point per workflow
  ) +
    geom_text(aes(y = mean - 0.2, label = wflow_id), angle = 90, hjust = 1) +
    lims(y = c(0.6, 1.4)) +
    theme(legend.position = "none")
  
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
  
  matched_results
  
  matched_results |>  
    ggplot(aes(x = complete, y = race)) + 
    geom_abline(lty = 3) + 
    geom_point() + 
    geom_text_repel(aes(label = model)) +
    coord_obs_pred() + 
    labs(x = "Complete Grid RMSE", y = "Racing RMSE") 
  
  # Create a list to store predictions for all 12 models
  all_model_predictions <- list()
  
  # Loop through the 12 models from matched_results and collect predictions
  for (i in seq_len(nrow(matched_results))) {
    
    # Extract the model information
    model_id <- matched_results$wflow_id[i]
    
    # Get the best configuration for the current model
    best_results <- race_results |>  
      extract_workflow_set_result(model_id) |> 
      select_best(metric = "rmse")
    
    # Finalize the model with the best hyperparameters
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
  ggplot(all_model_predictions_df, aes(x = log_BFA1000, y = .pred, color = dataset)) +
    geom_point(alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, color = "gray50", lty = 2) +  # Ideal line
    facet_wrap(~model, ncol = 4, nrow = 3) +  # Facet wrap for all models (4x3 grid)
    labs(title = "Predicted vs Observed Values for 12 Models",
         x = "Observed log_BFA1000",
         y = "Predicted log_BFA1000") +
    theme_bw() +
    scale_color_manual(values = c("calibration" = "blue", "verification" = "red"))
  
  #--------------
  # Ensemble mean
  if(metamodel == FALSE){
    models_stack <- 
      stacks() |>  
      add_candidates(race_results)
    
    models_stack
    
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
    
    View(rmse_results)
    
    set.seed(seed+1)
    ens <- blend_predictions(models_stack)
    
    autoplot(ens)
    
    set.seed(seed+2)
    ens <- blend_predictions(models_stack, penalty = 10^seq(-2, -0.5, length = 20))
    
    autoplot(ens)
    
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
    weights_labeled <- weights_data %>%
      left_join(lookup, by = "terms") %>%
      mutate(label = factor(label, levels = unique(label)))  # optional
    
    weights_labeled <- weights_labeled %>%
      arrange(desc(weight)) %>%
      mutate(member_rank = row_number())  # this will be used as y-axis
    
    stack_rank <- ggplot(weights_labeled, aes(x = weight, y = reorder(member_rank, weight))) +
      geom_col(aes(fill = label)) +
      geom_text(aes(x = weight + 0.01, label = label), hjust = 0) +
      theme_bw() +
      theme(legend.position = "none") +
      labs(
        title = paste0("Penalty = ", signif(ens$penalty$penalty, 3)),  # <- Penalty added here
        x = "Stacking Coefficient",
        y = "Member"
      ) +
      lims(x = c(0, 0.4))
    
  
    # Save the plot
    if(use_country == TRUE){
      plot_name <- paste0(out_path, "/stack_rank_ensemble_country.png")
    }else{
      plot_name <- paste0(out_path, "/stack_rank_ensemble.png")
    }
    ggsave(plot_name, plot = stack_rank, width = 1 * 140, height = 1 * 130, dpi = 600, units = 'mm')
    
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
  ggplot(ens_model_pred_df, aes(x = log_BFA1000, y = .pred, color = dataset)) +
    geom_point(alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, color = "gray50", lty = 2) +  # Ideal line
    labs(title = "Predicted vs Observed Values - Model Ensemble Mean",
         x = "Observed log_BFA1000",
         y = "Predicted log_BFA1000") +
    theme_bw() +
    scale_color_manual(values = c("calibration" = "blue", "verification" = "red")) +
    geom_text(data = metrics_values, aes(x = min(ens_model_pred_df$log_BFA1000), 
                                         y = y_position,
                                         label = metric_label),
              hjust = 0, color = "black", size = 4, parse = TRUE)
  
  # Prepare RMSE values with multiline text using ggtext styling
  rmse_values_combined <- metrics_values |> 
    select(dataset, rmse) |> 
    mutate(combined_label = paste0(
      "RMSE<sub>calibration</sub> = ", sprintf("%.2f", rmse[dataset == "calibration"]), "<br>",
      "RMSE<sub>verification</sub> = ", sprintf("%.2f", rmse[dataset == "verification"])
    ),
    y_position = 5.5
    )
  
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
  
  # Save the plot
  if(use_country == TRUE){
    plot_name <- paste0(out_path, "/ensemble_models_predictions_log-scale_country.png")
  }else{
    plot_name <- paste0(out_path,"/ensemble_models_predictions_log-scale.png")
  }
  ggsave(plot_name, plot = main_plot_ens_pred, width = 1 * 140, height = 1 * 130, dpi = 600, units = 'mm')
  
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
      y_position = 5.5
    )
  
  range(all_model_predictions_df$log_BFA1000)
  range(all_model_predictions_df$.pred)
  
  # Create the main plot with the legend
  num_countries <- length(unique(all_model_predictions_df$Country))
  main_plot_with_legend <- ggplot(all_model_predictions_df, aes(x = log_BFA1000, y = .pred, color = Country, shape = Country)) +
    geom_point(aes(fill = ifelse(dataset == "verification", Country, NA)), 
               size = 3, alpha = 0.8, stroke = 1) +  # stroke adjusts border width
    facet_wrap(~ model) +  # Create facets for each model
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
    rel_heights = c(0.1, 1, 0.1, 1, 1.5)  # Adjust these to control spacing/alignment
  )
  
  # Combine the two legends into one
  final_p <- plot_grid(main_plot,
                       legends,
                       nrow = 1,
                       align = "h",
                       axis = "t",
                       rel_widths = c(1, 0.1)
  ) + theme_bw() +
    theme(
      panel.border = element_blank()     # Remove the black border around the plot
    )
  
  print(final_p)
  
  # Save the plot
  if(use_country == TRUE){
    plot_name <- paste0(out_path, "/all_models_predictions_log-scale_country.png")
  }else{
    plot_name <- paste0(out_path, "/all_models_predictions_log-scale.png")
  }
  ggsave(plot_name, plot = final_p, width = 4 * 120, height = 3 * 110, dpi = 600, units = 'mm')
  
  #-------------------------------------------------------------------------------
  
  
  calibration_data |> 
    filter(Year %in% 1991:2020, Country == "CZE") |> 
    pull(log_BFA1000) |> 
    mean() |> 
    exp()
  
  # Load data
  data_scenarios <- read_csv("./inputs/BFAI_CZE_avg_indicators.csv")
  
  # Pivot longer to tidy format
  data_scenarios <- data_scenarios |> 
    pivot_longer(
      cols = -c(Variable, Season, Ccode), # Exclude the static columns
      names_to = "Scenario", 
      values_to = "Value"
    )
  
  # Extract components with corrected model names
  data_scenarios <- data_scenarios |> 
    mutate(
      SSP = str_extract(Scenario, "SSP\\d+"),                        # Extract 'SSP' followed by numbers
      SSP = str_replace_all(SSP, "SSP", ""),                         # Replace "SSP" from the name
      SSP = if_else(is.na(SSP), "Baseline", SSP),                    # Assign "Observed" for NA SSP
      Period = str_extract(Scenario, "\\d{4}(-\\d{4})?$"),           # Extract year or range
      Model = str_extract(Scenario, "(?<=SSP\\d{3}_)[a-z0-9\\-_]+"), # Extract model name
      Model = str_remove(Model, "_\\d{4}$"),                         # Remove year suffix from model
      Model = str_replace_all(Model, "_", "-"),                      # Replace "_" with "-" for consistency
      Model = if_else(is.na(Model), "Observed", Model)                # Assign "Observed" for NA SSP
    ) |> select(-Variable, -Season, -Scenario)
  
  data_scenarios_wide <- data_scenarios |> 
    pivot_wider(
      names_from = Ccode,
      values_from = Value
    )
  
  # Extract mean values for Pines, Conifers, Broadleaved and BFA1000 for 1991-2020 in CZE
  cze_summary <- clean_data |> 
    filter(Country == "CZE") |> 
    summarise(
      Pines = mean(Pines, na.rm = TRUE),
      Conifers = mean(Conifers, na.rm = TRUE),
      Broadleaved = mean(Broadleaved, na.rm = TRUE),
      BFA1000_1991_2020 = exp(mean(log_BFA1000[Year %in% 1991:2020], na.rm = TRUE))
    )
  
  # Add the extracted values to data_scenarios_wide
  data_scenarios_wide <- data_scenarios_wide |> 
    mutate(
      Country = "CZE",
      Pines = cze_summary$Pines,
      Conifers = cze_summary$Conifers,
      Broadleaved = cze_summary$Broadleaved,
      BFA1000_1991_2020 = cze_summary$BFA1000_1991_2020
    )
  
  # Loop through the 12 models from matched_results and collect predictions
  for (i in seq_len(nrow(matched_results))) {
    
    # Extract the model information
    model_id <- matched_results$wflow_id[i]
    
    # Get the best configuration for the current model
    best_results <- race_results |>  
      extract_workflow_set_result(model_id) |>  
      select_best(metric = "rmse")
    
    # Finalize the model with the best hyperparameters
    final_model <- race_results |> 
      extract_workflow(model_id) |>  
      finalize_workflow(best_results) |>  
      fit(data = calibration_data)
    
    tmp <- predict(final_model, new_data = data_scenarios_wide) |> 
      bind_cols(data_scenarios_wide) |> 
      mutate(Model_Type = model_id)
    
    # Combine predictions into a single dataframe
    if(i == 1){
      final_prediction <- tmp
    }else{
      final_prediction <- bind_rows(final_prediction, tmp)
    }
  }
  
  final_prediction <- final_prediction |> 
    mutate(predicted_BFA1000 = exp(.pred)) |> 
    relocate(predicted_BFA1000, .before = 2)
  
  final_prediction <- final_prediction |> 
    select(predicted_BFA1000, Period, Model, Model_Type, BFA1000_1991_2020) |> 
    rename(Climate_model = Model, ML_model = Model_Type)
  
  glimpse(final_prediction)
  
  # Map Observed to Baseline and add Period_num
  final_prediction <- final_prediction |>
    mutate(
      Climate_model = if_else(Climate_model == "Observed", "Baseline", Climate_model),
      Climate_model = factor(
        Climate_model,
        levels = c("Baseline", "cmcc-esm2", "ec-earth3", "gfdl-esm4",
                   "mpi-esm1-2-hr", "mri-esm2-0", "taiesm1")
      ),
      Period_num = case_when(
        Period == "1961-1990" ~ 1976,
        Period == "1981-2010" ~ 1996,
        Period == "1991-2020" ~ 2006,
        Period == "2030"      ~ 2030,
        Period == "2050"      ~ 2050,
        Period == "2070"      ~ 2070,
        Period == "2085"      ~ 2085,
        TRUE ~ NA_real_
      )
    )
  
  # Compute per-ML-model smoothed lines
  trend_data <- final_prediction |>
    mutate(period_type = if_else(Period %in% c("1961-1990", "1981-2010", "1991-2020"), "baseline", "future")) |>
    group_by(ML_model, period_type, Period_num) |>
    summarise(predicted_BFA1000 = if_else(
      period_type == "future",
      mean(predicted_BFA1000, na.rm = TRUE),
      predicted_BFA1000[1]  # use actual point from baseline
    ), .groups = "drop") |>
    group_by(ML_model) |>
    arrange(Period_num) |>
    group_modify(~{
      fit <- loess(predicted_BFA1000 ~ Period_num, data = .x, span = 0.8)
      smoothed <- data.frame(Period_num = seq(min(.x$Period_num), max(.x$Period_num), length.out = 100))
      smoothed$predicted_BFA1000 <- predict(fit, newdata = smoothed)
      smoothed$ML_model <- unique(.x$ML_model)
      smoothed
    }) |>
    ungroup()
  
  x_labels <- c(1970, 1990, 2010, 2030, 2050, 2070, 2090)
  
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
      aes(yintercept = BFA1000_1991_2020, linetype = "Observed (1991–2020)"),
      data = final_prediction |> filter(Period == "1991-2020"),
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
      limits = c(1968, 2092)
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
      values = c("Observed (1991–2020)" = "dashed", "Ensemble trend" = "solid"),
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
    ) +
    coord_cartesian(ylim = c(0.05, 0.35))
  
  plot_name <- if (use_country == TRUE) {
    paste0(out_path, "/all_models_predictions_for_scenarios_country.png")
  } else {
    paste0(out_path, "/all_models_predictions_for_scenarios.png")
  }
  ggsave(plot_name, plot = scenario_p, width = 3 * 140, height = 2 * 120, dpi = 600, units = 'mm')
  
  #-------------------------------------------------------------------------------
  ens_scen <- predict(ens, new_data = data_scenarios_wide) |> 
    bind_cols(data_scenarios_wide) |> 
    mutate(Model_Type = "ensemble mean")
  
  ens_scen <- ens_scen |> 
    mutate(predicted_BFA1000 = exp(.pred)) |> 
    relocate(predicted_BFA1000, .before = 2)
  
  ens_scen <- ens_scen |> 
    select(predicted_BFA1000, Period, Model, Model_Type, BFA1000_1991_2020) |> 
    rename(Climate_model = Model, ML_model = Model_Type)
  
  glimpse(ens_scen)
  
  # Update the Climate_model column to replace "Observed" with "Baseline"
  ens_scen <- ens_scen |> 
    mutate(Climate_model = ifelse(Climate_model == "Observed", "Baseline", Climate_model))
  
  clean_data |> filter(Year %in% 1991:2020) |> filter(Country == "CZE") |> pull(log_BFA1000) |> mean() |> exp()
  
  final_prediction |> filter(Period == "1991-2020") |> pull(BFA1000_1991_2020)
  
  # Compute period-wise mean for future periods
  mean_scenarios <- ens_scen |> 
    filter(Period %in% c("2030", "2050", "2070", "2085")) |> 
    group_by(Period) |> 
    summarize(predicted_BFA1000 = mean(predicted_BFA1000), .groups = "drop")
  
  # Get distinct baseline periods
  baseline_means <- ens_scen |> 
    filter(Period %in% c("1961-1990", "1981-2010", "1991-2020")) |> 
    select(Period, predicted_BFA1000) |> 
    distinct()
  
  # Combine for smoothing
  line_data <- bind_rows(baseline_means, mean_scenarios)
  
  # Define numeric positions for each period
  period_map <- c(
    "1961-1990" = 1976,
    "1981-2010" = 1996,
    "1991-2020" = 2006,
    "2030"      = 2030,
    "2050"      = 2050,
    "2070"      = 2070,
    "2085"      = 2085
  )
  
  # Apply mapping
  ens_scen <- ens_scen |> mutate(Period_num = period_map[Period])
  line_data <- line_data |> mutate(Period_num = period_map[Period])
  
  # Fit loess model for smoothed line
  loess_fit <- loess(predicted_BFA1000 ~ Period_num, data = line_data)
  smoothed_data <- data.frame(
    Period_num = seq(min(line_data$Period_num), max(line_data$Period_num), length.out = 100)
  )
  smoothed_data$predicted_BFA1000 <- predict(loess_fit, newdata = smoothed_data)
  
  # Define desired tick marks and labels for x-axis
  x_labels <- c(1970, 1990, 2010, 2030, 2050, 2070, 2090)
  
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
      aes(yintercept = BFA1000_1991_2020, linetype = "Observed (1991–2020)"),
      data = final_prediction |> filter(Period == "1991-2020"),
      color = "#2b2b2b", size = 0.8
    ) +
    
    # Model points on top
    geom_point(size = 3, alpha = 0.8) +
    
    # X axis scale
    scale_x_continuous(
      breaks = x_labels,
      labels = x_labels,
      limits = c(1968, 2092)
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
      values = c("Observed (1991–2020)" = "dashed", "Ensemble trend" = "solid"),
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
    plot_name <- paste0(out_path, "/all_models_predictions_for_scenarios_ensemble_mean_country.png")
  } else {
    plot_name <- paste0(out_path, "/all_models_predictions_for_scenarios_ensemble_mean.png")
  }
  ggsave(plot_name, plot = scenario_p, width = 140, height = 100, dpi = 600, units = 'mm')
  
  # Extract RMSEs from models
  rmse_out_all_models <- rmse_values_combined |>
    extract(combined_label,
            into = c("calibration", "verification"),
            regex = "calibration</sub> = ([0-9.]+)<br>RMSE<sub>verification</sub> = ([0-9.]+)",
            convert = TRUE) |>
    select(model, calibration, verification)
  
  # Extract and reshape ensemble RMSEs
  ensemble_row <- metrics_values |>
    extract(metric_label,
            into = c("calibration", "verification"),
            regex = "RMSE == ([0-9.]+) ~ .* R\\^2 == ([0-9.]+)",
            convert = TRUE) |>
    select(dataset, calibration) |>
    pivot_wider(names_from = dataset, values_from = calibration) |>
    mutate(model = "ensemble") |>
    select(model, calibration, verification)
  
  # Combine all RMSEs
  error_stat <- bind_rows(rmse_out_all_models, ensemble_row)
  
  write_csv(error_stat, paste0(out_path, "/error_stat.csv"))
}