# BFAI: machine-learning assessment of European burnt forest area

[![R](https://img.shields.io/badge/R-4.5.2-blue.svg)](https://www.r-project.org/)
[![workflow](https://img.shields.io/badge/workflow-tidymodels-blue.svg)](https://www.tidymodels.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17311553-blue.svg)](https://doi.org/10.5281/zenodo.17311553)

This repository contains the R code, input data, and workflow used for the manuscript:

> **Machine learning assessment of climate-driven variability in European forest fire burnt areas**  
> Emil Cienciala, Milan Fischer, Lucie Kudláčková, Markéta Poděbradská, Jan Balek, Radka Mašková, Petr Štěpánek, Jana Beranová, and Miroslav Trnka

The study evaluates whether climate-related fire-weather and drought predictors remain robust drivers of annual burnt forest area across Europe despite substantial changes in fire-management effectiveness. The central response variable is the **Burnt Forest Area Index (BFAI)**, expressed as annual burnt forest area per 1000 ha of national forest area.

The workflow is written in R and follows the modelling philosophy of [`tidymodels`](https://www.tidymodels.org/) and [_Tidy Modeling with R_](https://www.tmwr.org/) by Max Kuhn and Julia Silge. The archived research workflow is available through Zenodo: <https://doi.org/10.5281/zenodo.17311553>.

---

## Scientific overview

The analysis combines country-level forest-fire statistics, forest-resource information, climate-derived fire-risk indices, and machine-learning models to assess historical and future variability in BFAI.

The manuscript analyses **22 European countries** for **1990–2024**:

`AUT`, `BEL`, `BGR`, `CHE`, `CZE`, `DEU`, `DNK`, `ESP`, `FIN`, `FRA`, `GRC`, `HRV`, `ITA`, `LTU`, `LVA`, `NOR`, `POL`, `PRT`, `ROU`, `SVK`, `SVN`, and `SWE`.

The main scientific questions are:

1. Can climate-based statistical and machine-learning models robustly predict annual burnt forest area across contrasting European fire regimes?
2. Do observed BFAI trends indicate a continuing decoupling between climate pressure and realised burnt area, or is this decoupling weakening?
3. How might BFAI change under future climate conditions, especially under the SSP2-4.5 climate scenario?

Key results reported in the manuscript include:

- most countries showed stable or declining historical BFAI trends, despite increasingly fire-conducive conditions;
- a pan-European aggregate break-point around **2014** indicates a shift from a declining to an increasing aggregate BFAI trend;
- the final stacked ensemble explained approximately **79%** and **80%** of BFAI variance in calibration and verification datasets, respectively;
- annual vapour pressure deficit (`VPD_year`), country-specific effects, forest composition, Fire Weather Index variables, and available water deficit variables were among the most influential predictors;
- SSP2-4.5 scenario projections indicate moderate but consistent increases in BFAI for representative northern, central, and southern European countries.

---

## Repository structure

```text
BFAI/
├── README.md
├── LICENSE
├── run_all.R
├── pipeline_worker.R
├── pipeline_function.R
├── plot_scenarios_worker.R
├── inputs/
│   ├── BFAI_1990-2024.csv
│   ├── AnalyticalData2026.csv
│   ├── BFAI_CZE_avg_indicators.csv
│   ├── BFAI_inputs.xlsx
│   ├── InputData2026.xlsx
│   ├── baseline/
│   │   ├── *_country.csv
│   │   └── *_forest.csv
│   ├── scenariomip/
│   │   └── climate-scenario predictor files
│   └── scenariomip_50km/
│       └── higher-resolution scenario predictor files
├── src/
│   ├── load_libraries.R
│   ├── log_message.R
│   ├── files_manage.R
│   ├── colors.R
│   ├── corellation_filter.R
│   ├── boruta.R
│   ├── model_specification.R
│   ├── metafit_ens.R
│   ├── perturbation_audit.R
│   └── post_processing/
│       ├── all_predictions_check.R
│       ├── compare_GCMs.R
│       ├── correlation_matrix.R
│       ├── extract_scenario_ensemble_means.R
│       ├── extract_stacking_coefficients.R
│       ├── plot_scenario_ensemble_means.R
│       ├── select_model.R
│       ├── sensitivity_test.R
│       └── summarize.R
└── outputs/
    └── generated model outputs, figures, diagnostics, and summaries
```

Large generated model objects and scenario outputs are not intended to be tracked in Git. They are created locally under `outputs/` when the workflow is run.

---

## Main workflow files

The workflow is organised as a three-level execution cascade:

```text
run_all.R
└── pipeline_worker.R
    └── pipeline_function.R
        └── src/*.R helper scripts
```

| Path | Role |
| --- | --- |
| `run_all.R` | Main entry point. Defines the run grid, creates output folders, saves `runs.rds`, and launches each model run in a fresh R session using `callr::rscript()`. |
| `pipeline_worker.R` | Lightweight worker script. Reads the selected run index from command-line arguments, loads `runs.rds`, sources `pipeline_function.R`, and calls `run_pipeline()`. |
| `pipeline_function.R` | Core modelling pipeline. Loads libraries and helper functions, reads input data, performs filtering and feature selection, tunes candidate models, builds ensembles, runs robustness checks, and writes outputs. |
| `plot_scenarios_worker.R` | Worker script for scenario plotting/post-processing tasks. |
| `src/` | Modular helper scripts used by `pipeline_function.R`. |

This cascade is intentional. Each call from `run_all.R` starts a clean external R session via `callr`, which prevents memory accumulation between model configurations and gives each run a clean workspace. This is important because the workflow creates large tuning objects, fitted models, stacked ensembles, and scenario-prediction objects.

---

## Input data

### Main observed BFAI data

The main observed response data are stored in:

```text
inputs/BFAI_1990-2024.csv
```

This file contains annual country-level burnt forest area information used to derive or provide the Burnt Forest Area Index for the analysed period.

Additional input tables include:

```text
inputs/AnalyticalData2026.csv
inputs/BFAI_CZE_avg_indicators.csv
inputs/BFAI_inputs.xlsx
inputs/InputData2026.xlsx
```

### Baseline climate and forest predictors

Historical baseline predictors are stored in:

```text
inputs/baseline/
```

The pipeline reads the `*_forest.csv` files from this folder and reshapes them into the annual country-level predictor table used for model fitting. The corresponding `*_country.csv` files provide country-scale predictor information.

### Climate-scenario predictors

Scenario predictors are stored in:

```text
inputs/scenariomip/
inputs/scenariomip_50km/
```

These folders contain CMIP6 ScenarioMIP-based climate predictor files used for future BFAI projections. The workflow applies the trained models to scenario data for multiple SSPs, while the manuscript focuses on SSP2-4.5.

---

## Data and predictors

The response variable is:

```text
BFAI = annual burnt forest area / national forest area × 1000
```

where BFAI is expressed as ha burnt per kha of forest.

The predictor set combines climate, drought, fuel-moisture, fire-weather, forest-composition, and country-level information. The manuscript describes 18 base climate and land-surface variables aggregated over full-year and seasonal windows, producing 90 climate-related predictors before feature selection. Together with forest composition and country, this gives 94 initial candidate predictors.

Main predictor families include:

| Predictor family | Examples | Interpretation |
| --- | --- | --- |
| Fire Weather Index | `FWI_DI`, `FWI_3+`, `FWI_4+`, `FWI_5+`, `FWI_6` | Fire-weather intensity and counts of days above danger thresholds. |
| Dead fuel moisture | `DFMC10H`, `DFMC10H_u10`, `DFMC10H_u8`, etc. | Mean or threshold-based 10-hour dead fuel moisture indicators. |
| Vapour pressure deficit | `VPD` | Atmospheric moisture demand derived from temperature and relative humidity. |
| Soil-water status | `AWR`, `AWP`, `AWD` | Soil drought and water-deficit indicators derived from SoilClim. |
| Forest composition | conifers, broadleaves, pines | Time-invariant country-level vegetation composition. |
| Country | categorical predictor | Captures country-specific effects such as management, reporting, land use, and institutional differences. |

Seasonal aggregation windows include:

- full year;
- March–October (`MAMJJASO`);
- April–September (`AMJJAS`);
- October–March (`ONDJFM`);
- November–February (`NDJF`).

---

## Modelling workflow

The modelling workflow follows the tidy modelling approach implemented in [`tidymodels`](https://www.tidymodels.org/) and described in [_Tidy Modeling with R_](https://www.tmwr.org/). It uses recipes, workflows, model specifications, resampling, tuning, performance metrics, and stacked ensembles.

### 1. Run-grid definition

`run_all.R` defines a grid of modelling configurations, including:

- correlation-filter threshold;
- whether country is used as a predictor;
- whether meteorological and winter-season predictors are included;
- grid-search and racing-grid sizes;
- number of ensemble repetitions;
- perturbation and scenario-audit settings;
- number of CPU cores used for tuning and plotting.

Each row in this grid becomes a separate run with its own output folder:

```text
outputs/out_001/
outputs/out_002/
...
```

### 2. Clean-session execution

For each run, `run_all.R` creates the corresponding output folder and launches:

```r
callr::rscript(
  script = "pipeline_worker.R",
  cmdargs = as.character(run_i),
  stdout = log_file,
  stderr = log_file
)
```

This means each model run is evaluated in a separate R process. The design reduces cross-run memory leakage and avoids carrying large objects from one configuration into the next.

### 3. Data preparation

Inside `pipeline_function.R`, the workflow:

1. loads helper scripts from `src/`;
2. reads historical predictors from `inputs/baseline/`;
3. reads observed BFAI from `inputs/BFAI_1990-2024.csv`;
4. joins predictors and BFAI by `Country` and `Year`;
5. removes countries without suitable BFAI data;
6. cleans missing and infinite values;
7. prepares the log-transformed modelling target.

The model target is:

```text
log_BFA1000
```

which is the natural logarithm of BFAI.

### 4. Calibration and verification split

The annual observations are split using a repeated 3-year block structure:

- first and second years of each 3-year block: calibration;
- every third year: verification.

For the 1990–2024 period, this gives a temporally structured split while preserving observations across countries.

### 5. Feature selection

The workflow reduces the high-dimensional predictor set in three steps:

1. **Correlation filtering** removes multicollinear predictors.
2. **Boruta selection** identifies predictors that are consistently more informative than random shadow variables under cross-validation.
3. **Recursive feature elimination** ranks and retains predictors using Random Forest variable importance.

The final predictor set reported in the manuscript included:

```text
VPD_year
Country
Conifers
Pines
AWD0-40_sum_AMJJAS
FWI_3+_NDJF
FWI_4+_NDJF
AWD0-40_sum_NDJF
FWI_5+_ONDJFM
FWI_6_ONDJFM
```

### 6. Candidate models

The workflow evaluates 12 regression algorithms:

| Model family | Implemented models |
| --- | --- |
| Linear models | Generalized Linear Model, Elastic Net |
| Flexible regression | Multivariate Adaptive Regression Splines (MARS) |
| Support vector machines | Linear, polynomial, and radial kernels |
| Tree-based models | CART, bagged trees, Random Forest, Cubist |
| Boosting and Bayesian additive models | XGBoost, BART |
| Neural networks | Multilayer perceptron |

The model specifications are defined in:

```text
src/model_specification.R
```

using tidymodels-compatible model engines such as `glmnet`, `nnet`, `earth`, `kernlab`, `rpart`, `ranger`, `xgboost`, `Cubist`, and `dbarts`.

### 7. Preprocessing strategies

Each candidate model is combined with one or more preprocessing strategies, including:

- no preprocessing;
- normalization of numerical predictors;
- polynomial terms and interactions.

When `Country` is used, it is encoded through likelihood encoding using `step_lencode_glm()` rather than one-hot encoding.

### 8. Tuning and evaluation

Hyperparameters are tuned using:

- grid search;
- ANOVA racing via `finetune`.

Models are evaluated using 10-fold cross-validation on the calibration data and then independently assessed on the verification data. Main metrics include RMSE and coefficient of determination (`R²`).

### 9. Robustness and plausibility checks

The workflow includes perturbation and scenario-audit checks implemented through:

```text
src/perturbation_audit.R
```

These checks help screen candidate models or ensemble members for unrealistic behaviour under synthetic perturbations and high-end climate-scenario inputs.

### 10. Stacked ensemble

The final ensemble is built using stacked generalization through the `stacks` package. Predictions from tuned base learners are combined using penalized regression. The ensemble procedure is repeated with multiple random seeds to improve stability, and the final ensemble is selected based on calibration RMSE while preserving balanced calibration–verification performance.

---

## Climate-scenario projections

The project applies the trained ensemble model to climate-scenario predictors derived from CMIP6 global climate model simulations. The manuscript focuses on SSP2-4.5, although the workflow can process multiple SSPs.

Scenario processing uses the advanced delta-change approach, with a 1985–2014 baseline period and four future 30-year windows:

| Label | Period |
| --- | --- |
| 2030 | 2015–2044 |
| 2050 | 2035–2064 |
| 2070 | 2055–2084 |
| 2085 | 2070–2099 |

The manuscript presents scenario illustrations for Sweden, Czech Republic, and Italy, representing northern, central, and southern European conditions.

---

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/MilanFischer/BFAI.git
cd BFAI
```

### 2. Install R packages

The analysis was conducted in R 4.5.2. Install the core packages before running the workflow:

```r
install.packages(c(
  "tidyverse",
  "tidymodels",
  "stacks",
  "finetune",
  "yardstick",
  "recipes",
  "workflows",
  "workflowsets",
  "parsnip",
  "tune",
  "Boruta",
  "caret",
  "ranger",
  "randomForest",
  "xgboost",
  "Cubist",
  "earth",
  "kernlab",
  "kknn",
  "nnet",
  "dbarts",
  "baguette",
  "rules",
  "embed",
  "vip",
  "readxl",
  "yaml",
  "future",
  "furrr",
  "doFuture",
  "parallel",
  "callr",
  "ggrepel",
  "ggtext",
  "cowplot",
  "patchwork",
  "ragg",
  "RColorBrewer",
  "gtable",
  "lme4"
))
```

### 3. Run the full workflow

From the repository root, run:

```r
source("run_all.R")
```

or from a terminal:

```bash
Rscript run_all.R
```

The full workflow can be computationally demanding. It performs feature selection, repeated model tuning, racing, ensemble construction, robustness checks, scenario prediction, and post-processing. Runtime depends strongly on the number of configurations in `run_all.R` and on the number of CPU cores assigned to tuning and plotting.

### 4. Inspect logs and outputs

Each run writes outputs to a separate folder, for example:

```text
outputs/out_001/
```

The main log file for each run is:

```text
outputs/out_001/pipeline_log.txt
```

Typical generated outputs include:

- selected predictor configuration (`config.yml`);
- feature-importance plots;
- tuned model results;
- racing and grid-search results;
- fitted ensemble objects;
- perturbation-audit tables;
- observed-versus-predicted plots;
- scenario predictions by country and climate model;
- post-processed summary figures and tables.

Large generated `.rds` model objects should remain outside Git history.

---

## Post-processing

Selected post-processing scripts can be run after model outputs are available:

```r
source("src/post_processing/summarize.R")
source("src/post_processing/extract_stacking_coefficients.R")
source("src/post_processing/plot_scenario_ensemble_means.R")
```

Scenario plotting can also use the worker-based structure in:

```text
plot_scenarios_worker.R
```

---

## Reproducibility notes

- The workflow depends on stochastic procedures including resampling, hyperparameter tuning, Random Forest models, and stacked ensemble construction.
- Fixed seeds are used where possible, but small numerical differences can occur across operating systems, package versions, and parallel backends.
- `ragg` is used for stable cross-platform rendering of figures.
- Large generated files under `outputs/` are intentionally excluded from Git tracking.
- The repository is designed for scientific reproducibility of the manuscript analysis, not as a general-purpose R package.

---

## Methodological basis

This workflow is based on three pillars:

1. **tidymodels** — a collection of R packages for modelling and machine learning using tidyverse principles: <https://www.tidymodels.org/>.
2. **Tidy Modeling with R** — the practical reference for model workflows, resampling, tuning, recipes, model comparison, and ensembles: <https://www.tmwr.org/>.
3. **Archived research workflow** — Zenodo DOI: <https://doi.org/10.5281/zenodo.17311553>.

---

## Citation

If you use this repository, please cite the associated manuscript and software archive.

### Manuscript

Cienciala, E., Fischer, M., Kudláčková, L., Podebradská, M., Balek, J., Mašková, R., Štěpánek, P., Beranová, J., and Trnka, M.

*Machine learning assessment of climate-driven variability in European forest fire burnt areas.*

### Software and data repository

Fischer, M. (2025). *BFAI: machine-learning assessment of European burnt forest area* [Computer software and data].

GitHub repository: <https://github.com/MilanFischer/BFAI>

Zenodo archive: <https://doi.org/10.5281/zenodo.17311553>

### Methodological framework

This repository relies heavily on the **tidymodels** ecosystem and the workflow described in *Tidy Modeling with R*. When appropriate, please also cite:

```text
Kuhn, M., & Wickham, H. (2020).
tidymodels: a collection of packages for modeling and machine learning using tidyverse principles.
https://doi.org/10.21105/joss.02543

Kuhn, M., & Silge, J. (2022).
Tidy Modeling with R.
https://www.tmwr.org/
```

---

## License

This repository is distributed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for details.

---

## Contact

For questions about the code, data processing, or reproducibility workflow, please open an issue in this repository or contact the repository maintainer.