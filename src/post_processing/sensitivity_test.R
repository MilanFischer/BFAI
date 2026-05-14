# ============================================================
# Directional sensitivity / physical plausibility analysis
# Final stacked ensemble model (`ens`)
# ============================================================
#
# Purpose
# -------
# This analysis evaluates whether the final ensemble model responds
# to selected predictors in physically meaningful directions.
#
# A PDP-style (partial dependence style) sensitivity analysis is
# performed by varying one predictor across its observed range while
# keeping all remaining predictors unchanged, followed by prediction
# with the fitted stacked ensemble model.
#
# Local slopes of the response curves are then compared against
# expected physical directions derived from fire ecology and drought
# theory.
#
#
# Interpretation
# --------------
# expected_sign = +1
#   Increasing predictor should increase predicted BFAI
#
# expected_sign = -1
#   Increasing predictor should decrease predicted BFAI
#
# expected_sign = 0
#   No simple monotonic physical expectation
#
#
# Predictor definitions and expected directions
# ---------------------------------------------
#
# FWI_FD
#   Mean Fire Weather Index during a given period
#   Higher FWI = more severe fire weather
#   Expected direction: POSITIVE (+1)
#
# FWI_3+, FWI_4+, FWI_5+, FWI_6
#   Number of days exceeding given FWI thresholds
#   Higher values = more dangerous fire weather days
#   Expected direction: POSITIVE (+1)
#
# DFM10H
#   10-hour dead fuel moisture (%)
#   Higher moisture reduces flammability
#   Expected direction: NEGATIVE (-1)
#
# DFM10H_u10, DFM10H_u8, DFM10H_u6, DFM10H_u4, DFM10H_u2
#   Number of days with dead fuel moisture below threshold
#   Higher values = more dry fuel days
#   Expected direction: POSITIVE (+1)
#
# VPD
#   Vapor pressure deficit
#   Higher VPD = stronger atmospheric drought and fuel drying
#   Expected direction: POSITIVE (+1)
#
# AWP0-40_S2+, AWP0-40_S3+, AWP0-40_S4+
#   Number of days in drought categories S2+/S3+/S4+
#   within the 0-40 cm soil profile
#   Higher values = more severe soil drought occurrence
#   Expected direction: POSITIVE (+1)
#
# AWD0-40_sum
#   Cumulative available water deficit in 0-40 cm soil profile [mm]
#
#   IMPORTANT:
#   In the processed dataset AWD values are stored as NEGATIVE values.
#
#   Therefore:
#     more negative AWD = stronger drought deficit
#     numerically higher AWD = wetter conditions
#
#   Consequently, the expected NUMERICAL direction in the model is:
#   Expected direction: NEGATIVE (-1)
#
# AWR0-10_u30, AWR0-40_u30
#   Number of days with relative available soil water below 30%
#   Higher values = more dry-soil days
#   Expected direction: POSITIVE (+1)
#
#
# Notes
# -----
# 1. PDP-style analyses can be affected by correlated predictors.
#    Therefore, weaker or partially inconsistent responses may occur
#    for highly collinear drought variables.
#
# 2. Small local slopes are ignored using a tolerance threshold
#    to avoid interpreting numerical noise as meaningful direction.
#
# 3. Strong physical consistency is expected mainly for dominant
#    predictors driving the ensemble response.
#
# 4. Results should be interpreted together with predictor importance,
#    ensemble performance, and ecological plausibility.
#
# ============================================================

dir.create(file.path(out_path, "sensitivity"), showWarnings = FALSE, recursive = TRUE)

library(tidyverse)
library(yardstick)

# Tiny changes in PDP curves should not be interpreted as real direction
slope_tolerance <- 0.002

# ------------------------------------------------------------
# 1. Define expected physical direction for predictors
#    +1 = predictor should increase predicted BFAI
#    -1 = predictor should decrease predicted BFAI
#     0 = no fixed expected direction / categorical / uncertain
#
# Important:
# AWD is stored as negative deficit values, so numerically higher AWD
# means less deficit. Therefore the expected numerical sign is negative.
# ------------------------------------------------------------

expected_direction <- tibble(
  predictor = predictors,
  expected_sign = case_when(
    str_detect(predictor, "AWD")                         ~ -1,
    str_detect(predictor, "FWI|VPD|AWP|AWR.*u|DFMC.*u|DFM10H.*u") ~  1,
    str_detect(predictor, "DFMC10H$|DFM10H$|DFMC$")      ~ -1,
    str_detect(predictor, "Conifers|Pines|Broadleaved") ~  0,
    predictor %in% c("Country", "Year")                 ~  0,
    TRUE                                                 ~ NA_real_
  ),
  expected_interpretation = case_when(
    str_detect(predictor, "AWD") & expected_sign == -1 ~
      "Expected negative: AWD is stored as negative deficit values; numerically higher AWD means less drought",
    expected_sign ==  1 ~
      "Expected positive: higher fire danger/drought frequency should increase BFAI",
    expected_sign == -1 ~
      "Expected negative: higher moisture or numerically less drought should reduce BFAI",
    expected_sign ==  0 ~
      "No simple universal expected direction",
    is.na(expected_sign) ~
      "Expected direction not assigned"
  )
)

write_csv(
  expected_direction,
  file.path(out_path, "sensitivity", "expected_predictor_directions.csv")
)

# ------------------------------------------------------------
# 2. Helper: model prediction
# ------------------------------------------------------------

predict_ens <- function(model, new_data) {
  predict(model, new_data = new_data) |>
    dplyr::pull(.pred)
}

# ------------------------------------------------------------
# 3. PDP-style directional check
# ------------------------------------------------------------

make_pdp_direction <- function(var, data, model, n_grid = 25) {
  
  if (!is.numeric(data[[var]])) return(NULL)
  
  x_grid <- quantile(
    data[[var]],
    probs = seq(0.05, 0.95, length.out = n_grid),
    na.rm = TRUE,
    names = FALSE
  ) |>
    unique()
  
  pdp <- map_dfr(x_grid, function(xval) {
    nd <- data
    nd[[var]] <- xval
    
    tibble(
      predictor = var,
      x = xval,
      mean_pred_log_BFAI = mean(predict_ens(model, nd), na.rm = TRUE)
    )
  })
  
  pdp |>
    arrange(x) |>
    mutate(
      local_slope = mean_pred_log_BFAI - dplyr::lag(mean_pred_log_BFAI),
      local_direction_raw = sign(local_slope),
      local_direction = case_when(
        is.na(local_slope) ~ NA_real_,
        abs(local_slope) < slope_tolerance ~ 0,
        TRUE ~ sign(local_slope)
      )
    )
}

pdp_direction <- map_dfr(
  predictors,
  make_pdp_direction,
  data = verification_data,
  model = ens,
  n_grid = 25
)

pdp_direction <- pdp_direction |>
  left_join(expected_direction, by = "predictor") |>
  mutate(
    physically_consistent = case_when(
      is.na(expected_sign) | expected_sign == 0 ~ NA,
      is.na(local_direction)                   ~ NA,
      local_direction == 0                     ~ NA,
      local_direction == expected_sign         ~ TRUE,
      TRUE                                     ~ FALSE
    )
  )

write_csv(
  pdp_direction,
  file.path(out_path, "sensitivity", "pdp_directional_sensitivity_raw.csv")
)

# ------------------------------------------------------------
# 4. Summary table
# ------------------------------------------------------------

direction_summary <- pdp_direction |>
  group_by(predictor, expected_sign, expected_interpretation) |>
  summarise(
    n_local_slopes_total = sum(!is.na(local_slope)),
    n_local_slopes_used = sum(!is.na(physically_consistent)),
    n_flat_or_tiny_slopes = sum(local_direction == 0, na.rm = TRUE),
    n_consistent = sum(physically_consistent, na.rm = TRUE),
    prop_consistent = ifelse(
      n_local_slopes_used > 0,
      n_consistent / n_local_slopes_used,
      NA_real_
    ),
    mean_slope = mean(local_slope, na.rm = TRUE),
    min_slope = min(local_slope, na.rm = TRUE),
    max_slope = max(local_slope, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(prop_consistent)

write_csv(
  direction_summary,
  file.path(out_path, "sensitivity", "directional_sensitivity_summary.csv")
)

print(direction_summary)

# ------------------------------------------------------------
# 5. Plot PDP directional curves
# ------------------------------------------------------------

p_pdp <- pdp_direction |>
  left_join(direction_summary |> select(predictor, prop_consistent), by = "predictor") |>
  mutate(
    predictor_label = paste0(
      predictor,
      "\nConsistency = ",
      ifelse(is.na(prop_consistent), "NA", round(prop_consistent, 2))
    )
  ) |>
  ggplot(aes(x = x, y = mean_pred_log_BFAI)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  facet_wrap(~ predictor_label, scales = "free_x") +
  theme_bw() +
  labs(
    title = "Directional sensitivity of final ensemble model",
    subtitle = paste0(
      "PDP-style check; slopes with absolute change < ",
      slope_tolerance,
      " ignored as near-flat"
    ),
    x = "Predictor value",
    y = "Mean predicted log(BFAI)"
  )

ggsave(
  file.path(out_path, "sensitivity", "pdp_directional_sensitivity.png"),
  p_pdp,
  width = 220,
  height = 160,
  units = "mm",
  dpi = 600
)

# ------------------------------------------------------------
# 6. ICE-style country-specific curves
# ------------------------------------------------------------

make_country_ice <- function(var, data, model, n_grid = 20) {
  
  if (!is.numeric(data[[var]])) return(NULL)
  if (!"Country" %in% names(data)) return(NULL)
  
  x_grid <- quantile(
    data[[var]],
    probs = seq(0.05, 0.95, length.out = n_grid),
    na.rm = TRUE,
    names = FALSE
  ) |>
    unique()
  
  map_dfr(x_grid, function(xval) {
    nd <- data
    nd[[var]] <- xval
    nd$.pred <- predict_ens(model, nd)
    
    nd |>
      group_by(Country) |>
      summarise(
        mean_pred_log_BFAI = mean(.pred, na.rm = TRUE),
        .groups = "drop"
      ) |>
      mutate(
        predictor = var,
        x = xval
      )
  })
}

country_ice <- map_dfr(
  predictors,
  make_country_ice,
  data = verification_data,
  model = ens,
  n_grid = 20
)

write_csv(
  country_ice,
  file.path(out_path, "sensitivity", "country_ice_directional_sensitivity.csv")
)

p_ice <- country_ice |>
  ggplot(aes(x = x, y = mean_pred_log_BFAI, group = Country, color = Country)) +
  geom_line(linewidth = 0.6, alpha = 0.8) +
  facet_wrap(~ predictor, scales = "free_x") +
  theme_bw() +
  labs(
    title = "Country-specific ICE-style directional sensitivity",
    x = "Predictor value",
    y = "Mean predicted log(BFAI)"
  )

ggsave(
  file.path(out_path, "sensitivity", "country_ice_directional_sensitivity.png"),
  p_ice,
  width = 240,
  height = 170,
  units = "mm",
  dpi = 600
)

# ------------------------------------------------------------
# 7. Flag potentially problematic predictors
# ------------------------------------------------------------

problem_predictors <- direction_summary |>
  filter(!is.na(prop_consistent), prop_consistent < 0.6)

write_csv(
  problem_predictors,
  file.path(out_path, "sensitivity", "potentially_problematic_predictors.csv")
)

print(problem_predictors)