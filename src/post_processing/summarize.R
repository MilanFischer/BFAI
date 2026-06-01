# ============================================================
# Summarize BFAI experiment outputs into PDF reports
# ============================================================

library(tidyverse)
library(rmarkdown)
library(glue)
library(yaml)

# ----------------
# User settings
# ----------------

project_root <- "d:/Git/GitHub/GitHubRepositories/BFAI"

outputs_dir <- file.path(project_root, "outputs")
summary_dir <- file.path(outputs_dir, "summary")

# Number of scenario plots placed on one page.
# 4 = usually one country per page with four SSPs.
# 16 = more compact, but smaller figures.
scenario_plots_per_page <- 4

# If TRUE, only outputs/out_001 ... outputs/out_012 are summarized.
# If FALSE, all outputs/out_* folders are summarized.
use_fixed_experiment_range <- TRUE
fixed_experiment_ids <- sprintf("out_%03d", 1:12)

# ----------------
# Normalize project root
# ----------------

project_root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)
outputs_dir  <- file.path(project_root, "outputs")
summary_dir  <- file.path(outputs_dir, "summary")

dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------
# TinyTeX / pdflatex check
# ----------------

if (Sys.which("pdflatex") == "") {
  if (requireNamespace("tinytex", quietly = TRUE) && tinytex::is_tinytex()) {
    pdflatex_path <- list.files(
      tinytex::tinytex_root(),
      pattern = "pdflatex.exe",
      recursive = TRUE,
      full.names = TRUE
    )

    if (length(pdflatex_path) > 0) {
      tex_bin <- dirname(pdflatex_path[1])
      Sys.setenv(
        PATH = paste(tex_bin, Sys.getenv("PATH"), sep = .Platform$path.sep)
      )
    }
  }
}

if (Sys.which("pdflatex") == "") {
  stop(
    "pdflatex not found. Install TinyTeX first, then restart RStudio:\n",
    "install.packages('tinytex')\n",
    "tinytex::install_tinytex()"
  )
}

# ----------------
# Helpers
# ----------------

safe_path <- function(path) {
  normalizePath(path, winslash = "/", mustWork = FALSE)
}

relative_path <- function(path, root = project_root) {
  path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  sub(paste0("^", root, "/?"), "", path)
}

read_text_file <- function(path) {
  if (!file.exists(path)) {
    return("File not found.")
  }

  paste(
    readLines(path, warn = FALSE, encoding = "UTF-8"),
    collapse = "\n"
  )
}

escape_backticks <- function(x) {
  str_replace_all(x, "```", "` ` `")
}

# Robust reader for config.yml field: use_country
# Accepts TRUE/FALSE, yes/no, y/n, 1/0.
read_use_country <- function(config_path) {
  if (!file.exists(config_path)) {
    return(NA)
  }

  cfg <- yaml::read_yaml(config_path)
  value <- cfg$use_country

  if (is.null(value)) {
    return(NA)
  }

  if (is.logical(value)) {
    return(isTRUE(value))
  }

  value_chr <- str_to_lower(str_trim(as.character(value)))

  if (value_chr %in% c("yes", "y", "true", "t", "1")) {
    return(TRUE)
  }

  if (value_chr %in% c("no", "n", "false", "f", "0")) {
    return(FALSE)
  }

  NA
}

overview_plot_names <- function(use_country) {
  suffix <- if (isTRUE(use_country)) "_country" else ""

  c(
    "feature_importance.png",
    paste0("all_models_predictions_log-scale", suffix, ".png"),
    paste0("stack_rank_ensemble", suffix, ".png"),
    paste0("ensemble_models_predictions_log-scale", suffix, ".png")
  )
}

scenario_file_pattern <- function(use_country) {
  suffix <- if (isTRUE(use_country)) "_country" else ""

  paste0(
    "^scenarios_[A-Z]{3}_ssp[0-9]+_",
    "all_models_predictions_for_scenarios_ensemble_mean",
    suffix,
    "\\.png$"
  )
}

make_image_chunk <- function(img, width = "95%") {
  img <- safe_path(img)

  glue(
    "\n```{{r, echo=FALSE, results='asis', out.width='{width}', fig.align='center'}}\n",
    "knitr::include_graphics('{img}')\n",
    "```\n"
  )
}

make_scenario_chunk <- function(paths, plots_per_page) {
  paths <- safe_path(paths)

  width <- case_when(
    plots_per_page <= 4 ~ "48%",
    plots_per_page <= 9 ~ "31%",
    TRUE ~ "23%"
  )

  paths_r <- paste0("'", paths, "'", collapse = ",\n  ")

  glue(
    "\n```{{r, echo=FALSE, results='asis', fig.show='hold', out.width='{width}', fig.align='center'}}\n",
    "knitr::include_graphics(c(\n  {paths_r}\n))\n",
    "```\n"
  )
}

extract_scenario_info <- function(path) {
  fname <- basename(path)

  m <- str_match(
    fname,
    "^scenarios_([A-Z]{3})_(ssp[0-9]+)_all_models_predictions_for_scenarios_ensemble_mean(?:_country)?\\.png$"
  )

  tibble(
    path = path,
    country = m[, 2],
    ssp = m[, 3]
  )
}

# ----------------
# Find experiments
# ----------------

experiment_dirs <- list.dirs(
  outputs_dir,
  recursive = FALSE,
  full.names = TRUE
) |>
  keep(\(x) str_detect(basename(x), "^out_[0-9]+$")) |>
  sort()

if (use_fixed_experiment_range) {
  experiment_dirs <- experiment_dirs |>
    keep(\(x) basename(x) %in% fixed_experiment_ids)
}

if (length(experiment_dirs) == 0) {
  stop("No experiment folders found in: ", outputs_dir)
}

# ----------------
# Build PDFs
# ----------------

for (exp_dir in experiment_dirs) {

  exp_name <- basename(exp_dir)
  message("Creating summary for: ", exp_name)

  config_path <- file.path(exp_dir, "config.yml")
  config_text <- read_text_file(config_path) |>
    escape_backticks()

  use_country <- read_use_country(config_path)

  if (is.na(use_country)) {
    warning(
      "Could not read use_country from ",
      relative_path(config_path),
      ". Assuming use_country = TRUE."
    )
    use_country <- TRUE
  }

  overview_plots_exp <- overview_plot_names(use_country)

  rmd_path <- file.path(summary_dir, paste0(exp_name, "_summary.Rmd"))
  pdf_path <- file.path(summary_dir, paste0(exp_name, "_summary.pdf"))

  rmd <- c(
    "---",
    glue("title: 'BFAI experiment summary: {exp_name}'"),
    "output:",
    "  pdf_document:",
    "    toc: true",
    "    toc_depth: 2",
    "    number_sections: true",
    "geometry: margin=1.5cm",
    "---",
    "",
    "```{r setup, include=FALSE}",
    "knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)",
    "```",
    "",
    "# Experiment",
    "",
    glue("Experiment: **{exp_name}**"),
    "",
    glue("Experiment folder: `{relative_path(exp_dir)}`"),
    "",
    glue("use_country: `{if_else(use_country, 'yes', 'no')}`"),
    "",
    glue("Created: `{Sys.Date()}`"),
    "",
    "\\newpage",
    "",
    "# Configuration",
    "",
    "```yaml",
    config_text,
    "```",
    "",
    "\\newpage"
  )

  # ----------------
  # Overview plots
  # ----------------

  overview_tbl <- tibble(
    label = overview_plots_exp,
    path = file.path(exp_dir, overview_plots_exp),
    exists = file.exists(file.path(exp_dir, overview_plots_exp))
  )

  for (i in seq_len(nrow(overview_tbl))) {
    rmd <- c(
      rmd,
      glue("# {overview_tbl$label[i]}"),
      ""
    )

    if (overview_tbl$exists[i]) {
      rmd <- c(
        rmd,
        make_image_chunk(overview_tbl$path[i], width = "95%")
      )
    } else {
      rmd <- c(
        rmd,
        glue("Missing file: `{relative_path(overview_tbl$path[i])}`")
      )
    }

    rmd <- c(rmd, "", "\\newpage")
  }

  # ----------------
  # Scenario plots
  # ----------------

  scenario_dir <- file.path(exp_dir, "scenarios")

  scenario_files <- list.files(
    scenario_dir,
    pattern = scenario_file_pattern(use_country),
    full.names = TRUE
  )

  rmd <- c(
    rmd,
    "# Scenario plots",
    "",
    glue("Scenario folder: `{relative_path(scenario_dir)}`"),
    ""
  )

  if (length(scenario_files) == 0) {
    rmd <- c(
      rmd,
      glue("No scenario plots found in `{relative_path(scenario_dir)}`.")
    )
  } else {
    scenario_tbl <- map_dfr(scenario_files, extract_scenario_info) |>
      filter(!is.na(country), !is.na(ssp)) |>
      mutate(
        ssp = factor(ssp, levels = c("ssp126", "ssp245", "ssp370", "ssp585"))
      ) |>
      arrange(country, ssp)

    for (country_i in sort(unique(scenario_tbl$country))) {
      country_tbl <- scenario_tbl |>
        filter(country == country_i) |>
        arrange(ssp)

      available_ssps <- paste(str_to_upper(as.character(country_tbl$ssp)), collapse = " | ")

      rmd <- c(
        rmd,
        glue("## {country_i}"),
        "",
        glue("**{available_ssps}**"),
        "",
        make_scenario_chunk(country_tbl$path, scenario_plots_per_page),
        "",
        "\\newpage"
      )
    }
  }

  writeLines(rmd, rmd_path, useBytes = TRUE)

  render(
    input = rmd_path,
    output_file = basename(pdf_path),
    output_dir = summary_dir,
    quiet = FALSE,
    clean = TRUE
  )
}

message("Done. PDFs written to: ", summary_dir)
