library(tidyverse)

source("./src/colors.R")

root <- "./outputs"
out_id <- "out_002"
out_path <- file.path(root, out_id)

ssp_use <- "ssp245"

countries_use <- c(
  SWE = "Sweden",
  CZE = "Czech Republic",
  # ITA = "Italy"
  HRV = "Croatia"
)

clean_data <- readRDS(file.path(out_path, "clean_data_for_plots.rds"))

make_country_panel_data <- function(country_code) {
  
  bfa_reference <- clean_data |>
    filter(Country == country_code, Year %in% 1991:2024) |>
    summarise(BFA1000_1991_2024 = exp(mean(log_BFA1000, na.rm = TRUE))) |>
    pull(BFA1000_1991_2024)
  
  ens_scen_annual <- readRDS(
    file.path(out_path, "plot_data", paste0(country_code, "_ensemble_predictions.rds"))
  )
  
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
    rename(Climate_model = GCM)
  
  baseline_ens <- clean_data |>
    filter(Country == country_code) |>
    mutate(
      Period = case_when(
        Year %in% 1981:2010 ~ "1981-2010",
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
      country = country_code,
      country_name = recode(country_code, !!!countries_use),
      Climate_model = factor(
        Climate_model,
        levels = c(
          "Baseline", "cmcc-esm2", "ec-earth3", "gfdl-esm4",
          "mpi-esm1-2-hr", "mri-esm2-0", "taiesm1"
        )
      ),
      Period_num = case_when(
        Period == "1981-2010" ~ 1996,
        Period == "2030"      ~ 2030,
        Period == "2050"      ~ 2050,
        Period == "2070"      ~ 2070,
        Period == "2085"      ~ 2085,
        TRUE ~ NA_real_
      )
    )
  
  mean_scenarios <- ens_scen |>
    filter(Period %in% c("2030", "2050", "2070", "2085")) |>
    group_by(country, country_name, Period, Period_num) |>
    summarise(
      predicted_BFA1000 = mean(predicted_BFA1000, na.rm = TRUE),
      .groups = "drop"
    )
  
  baseline_means <- ens_scen |>
    filter(Period == "1981-2010") |>
    select(country, country_name, Period, Period_num, predicted_BFA1000) |>
    distinct()
  
  line_data <- bind_rows(baseline_means, mean_scenarios) |>
    filter(is.finite(predicted_BFA1000), is.finite(Period_num))
  
  smoothed_data <- tibble(
    Period_num = seq(min(line_data$Period_num), max(line_data$Period_num), length.out = 100)
  )
  
  loess_fit <- loess(predicted_BFA1000 ~ Period_num, data = line_data)
  
  smoothed_data <- smoothed_data |>
    mutate(
      predicted_BFA1000 = predict(loess_fit, newdata = smoothed_data),
      country = country_code,
      country_name = recode(country_code, !!!countries_use)
    )
  
  list(
    points = ens_scen,
    smooth = smoothed_data
  )
}

plot_data <- map(names(countries_use), make_country_panel_data)

ens_points <- map_dfr(plot_data, "points")
smooth_lines <- map_dfr(plot_data, "smooth")

country_order <- unname(countries_use)

ens_points <- ens_points |>
  mutate(country_name = factor(country_name, levels = country_order))

smooth_lines <- smooth_lines |>
  mutate(country_name = factor(country_name, levels = country_order))

x_labels <- c(1990, 2010, 2030, 2050, 2070, 2090)

p <- ggplot(
  ens_points,
  aes(x = Period_num, y = predicted_BFA1000, color = Climate_model)
) +
  geom_line(
    data = smooth_lines,
    aes(x = Period_num, y = predicted_BFA1000, linetype = "Ensemble trend"),
    color = "#2b2b2b",
    size = 1.2,
    alpha = 0.3,
    inherit.aes = FALSE
  ) +
  geom_hline(
    aes(yintercept = BFA1000_1991_2024, linetype = "Observed\n(1991–2024)\nreference"),
    data = ens_points |> filter(Period == "1981-2010"),
    color = "#2b2b2b",
    size = 0.8
  ) +
  geom_point(size = 3, alpha = 0.8) +
  facet_wrap(~ country_name, ncol = 1, scales = "free_y") +
  scale_x_continuous(
    breaks = x_labels,
    labels = x_labels,
    limits = c(1988, 2092)
  ) +
  scale_color_manual(
    values = custom_colors_GCM,
    breaks = c(
      "Baseline", "cmcc-esm2", "ec-earth3", "gfdl-esm4",
      "mpi-esm1-2-hr", "mri-esm2-0", "taiesm1"
    ),
    name = "Climate data",
    guide = guide_legend(order = 1)
  ) +
  scale_linetype_manual(
    values = c(
      "Observed\n(1991–2024)\nreference" = "dashed",
      "Ensemble trend" = "solid"
    ),
    name = NULL,
    guide = guide_legend(
      override.aes = list(
        color = "#2b2b2b",
        size = c(0.8, 1.2),
        alpha = c(1, 0.3)
      ),
      order = 2,
      keywidth = 1.8,
      keyheight = unit(10, "mm")
    )
  ) +
  labs(
    x = "Year",
    y = "Predicted BFAI"
  ) +
  theme_bw() +
  theme(
    legend.position = "right",
    legend.text = element_text(
      size = 9,
      lineheight = 1.1
    ),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x = element_text(margin = margin(t = 15)),
    strip.text = element_text(face = "bold")
  )


country_string <- paste(names(countries_use), collapse = "_")

file_name <- paste(
  out_id,
  "scenarios",
  paste(names(countries_use), collapse = "_"),
  ssp_use,
  "ensemble_mean",
  sep = "_"
)

ggsave(
  filename = file.path(
    out_path,
    paste0(file_name, ".png")
  ),
  plot = p,
  width = 140,
  height = 240,
  dpi = 600,
  units = "mm",
  device = ragg::agg_png
)