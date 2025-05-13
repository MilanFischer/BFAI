get_boruta_predictors <- function(data, target, use_country = TRUE, use_year = FALSE, min_freq = 5, v = 10, seed = 1234) {
  set.seed(seed)
  
  # Step 1: Create formula based on input data
  exclude_cols <- c(target)
  if (!use_year) {
    exclude_cols <- c(exclude_cols, "Year")
  }
  if (!use_country) {
    exclude_cols <- c(exclude_cols, "Country")
  }
  
  # Determine predictors
  predictors <- setdiff(names(data), exclude_cols)
  
  formula <- as.formula(paste(target, "~", paste(predictors, collapse = " + ")))
  
  # Step 2: Create cross-validation folds
  cv_folds <- vfold_cv(data, v = v, strata = all_of(target))
  
  # Step 3: Run Boruta on each training split
  boruta_vars_by_fold <- purrr::map(cv_folds$splits, function(split) {
    training_data <- analysis(split)
    
    rec <- recipe(formula, data = training_data) |>
      step_nzv(all_predictors())
    
    if (use_country) {
      rec <- rec |>
        step_lencode_glm(Country, outcome = target)
    }
    
    rec <- rec |>
      step_select_boruta(all_predictors(), outcome = target)
    
    prep(rec, training = training_data, verbose = FALSE) |>
      bake(new_data = NULL) |>
      names()
  })
  
  # Step 4: Count frequency of each selected variable
  all_vars <- unlist(boruta_vars_by_fold)
  var_freq <- tibble::as_tibble(table(all_vars)) |>
    dplyr::arrange(dplyr::desc(n))
  
  # Step 5: Return only variables selected in at least `min_freq` folds
  final_sel <- var_freq |>
    dplyr::filter(n >= min_freq) |>
    dplyr::pull(all_vars) |>
    as.character()
  
  return(final_sel)
}
