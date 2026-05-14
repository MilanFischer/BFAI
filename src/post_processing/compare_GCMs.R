# ============================================================
# Compare climate model forcing variables across GCMs and periods
# ============================================================

library(tidyverse)

# ----------------
# User settings
# ----------------

scenario_root <- "./inputs"
country_code  <- "CZE"

# Use "forest" or "country"
file_type <- "forest"

# Optional: set to NULL to keep all scenarios
selected_scenario <- NULL
# selected_scenario <- "ssp245"

# Output folder
out_dir <- "./outputs/model_forcing_comparison"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------
# Load scenario data
# ----------------

scenario_country_files <- list.files(
  path = scenario_root,
  pattern = paste0(country_code, "_", file_type, "\\.csv$"),
  recursive = TRUE,
  full.names = TRUE
) |>
  keep(\(x) str_detect(x, "scenariomip"))

scenario_forcing_long <- scenario_country_files |>
  map_dfr(\(file) {
    
    # Works with both Windows "\" and Unix "/" path separators
    path_parts <- str_split(file, "[/\\\\]", simplify = TRUE)
    
    read_csv(file, show_col_types = FALSE) |>
      pivot_longer(
        cols = matches("^\\d{4}$"),
        names_to = "Year",
        values_to = "value"
      ) |>
      mutate(
        Country  = country_code,
        FileType = file_type,
        Dataset  = "scenario",
        Scenario = path_parts[length(path_parts) - 3],
        GCM      = path_parts[length(path_parts) - 2],
        Period   = as.numeric(path_parts[length(path_parts) - 1]),
        Year     = as.numeric(Year),
        varname  = paste(Variable, Season, sep = "_")
      ) |>
      select(Country, FileType, Scenario, GCM, Period, Year, Dataset, varname, value)
  }) |>
  filter(!is.na(value))

# Optional scenario filtering
if (!is.null(selected_scenario)) {
  scenario_forcing_long <- scenario_forcing_long |>
    filter(Scenario == selected_scenario)
}

# Save extracted forcing data
write_csv(
  scenario_forcing_long,
  file.path(out_dir, paste0(country_code, "_", file_type, "_scenario_forcing_long.csv"))
)

# ----------------
# Diagnostic check for SRAD / GFDL
# ----------------

srad_gfdl_check <- scenario_forcing_long |>
  filter(
    GCM == "gfdl-esm4",
    str_detect(varname, "^SRAD")
  ) |>
  group_by(Scenario, GCM, Period, varname) |>
  summarise(
    n = n(),
    min = min(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    max = max(value, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(Scenario, Period, varname)

print(srad_gfdl_check, n = Inf)

write_csv(
  srad_gfdl_check,
  file.path(out_dir, paste0(country_code, "_", file_type, "_SRAD_gfdl_check.csv"))
)

# ----------------
# Basic summary table
# ----------------

forcing_summary <- scenario_forcing_long |>
  group_by(Country, FileType, Scenario, GCM, Period, varname) |>
  summarise(
    n = n(),
    mean = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    min = min(value, na.rm = TRUE),
    max = max(value, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  forcing_summary,
  file.path(out_dir, paste0(country_code, "_", file_type, "_scenario_forcing_summary.csv"))
)

# ----------------
# Boxplots: one plot per forcing variable
# ----------------

plot_one_variable <- function(var) {
  
  plot_data <- scenario_forcing_long |>
    filter(varname == var) |>
    mutate(
      Period = factor(Period),
      GCM = factor(GCM)
    )
  
  p <- ggplot(plot_data, aes(x = GCM, y = value, fill = Period)) +
    geom_boxplot(outlier.alpha = 0.4) +
    facet_wrap(~ Scenario, scales = "free_x") +
    labs(
      title = paste("Model forcing comparison:", var),
      subtitle = paste("Country:", country_code, "| File type:", file_type),
      x = "Climate model",
      y = var,
      fill = "Period"
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold")
    )
  
  ggsave(
    filename = file.path(out_dir, paste0(country_code, "_", file_type, "_boxplot_", var, ".png")),
    plot = p,
    width = 11,
    height = 6,
    dpi = 300
  )
  
  return(p)
}

variables_to_plot <- scenario_forcing_long |>
  distinct(varname) |>
  arrange(varname) |>
  pull(varname)

plots <- map(variables_to_plot, plot_one_variable)

# ----------------
# SRAD-only plot
# ----------------

p_srad <- scenario_forcing_long |>
  filter(str_detect(varname, "^SRAD")) |>
  mutate(
    Period = factor(Period),
    GCM = factor(GCM)
  ) |>
  ggplot(aes(x = GCM, y = value, fill = Period)) +
  geom_boxplot(outlier.alpha = 0.4) +
  facet_grid(varname ~ Scenario, scales = "free_y") +
  labs(
    title = paste(country_code, file_type, "SRAD comparison"),
    x = "Climate model",
    y = "SRAD",
    fill = "Period"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold")
  )

# ggsave(
#   filename = file.path(out_dir, paste0(country_code, "_", file_type, "_SRAD_boxplots.png")),
#   plot = p_srad,
#   width = 14,
#   height = 8,
#   dpi = 300
# )

