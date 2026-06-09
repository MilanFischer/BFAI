# BFAI: machine-learning assessment of European burnt forest area

[![R](https://img.shields.io/badge/R-4.5.2-blue.svg)](https://www.r-project.org/)
[![tidymodels](https://img.shields.io/badge/workflow-tidymodels-blue.svg)](https://www.tidymodels.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17311553.svg)](https://doi.org/10.5281/zenodo.17311553)

This repository contains the R code, input data, and supporting scripts used for the manuscript:

> **Machine learning assessment of climate-driven variability in European forest fire burnt areas**  
> Emil Cienciala, Milan Fischer, Lucie Kudláčková, Markéta Podebradská, Jan Balek, Radka Mašková, Petr Štěpánek, Jana Beranová, and Miroslav Trnka

The study evaluates whether climate-related fire-weather and drought predictors remain robust drivers of annual burnt forest area across Europe despite substantial changes in fire-management effectiveness. The central response variable is the **Burnt Forest Area Index (BFAI)**, defined as annual burnt forest area per 1000 ha of national forest area.

The workflow is written in R and follows the modelling philosophy of [`tidymodels`](https://www.tidymodels.org/) and the book [_Tidy Modeling with R_](https://www.tmwr.org/) by Max Kuhn and Julia Silge. The repository is also archived through Zenodo: <https://doi.org/10.5281/zenodo.17311553>.

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
- the final stacked ensemble explained a substantial proportion of BFAI variance in both calibration and verification datasets;
- annual vapour pressure deficit (`VPD_year`), country-specific effects, forest composition, Fire Weather Index variables, and available water deficit variables were among the most influential predictors;
- SSP2-4.5 scenario projections indicate moderate but consistent increases in BFAI for representative northern, central, and southern European countries.

---

## Repository structure

```text
BFAI/
├── README.md
├── LICENSE
├── BFAI_model_MAIN.R
├── inputs/
│   ├── AnalyticalData2026.csv
│   ├── BFAI_inputs.xlsx
│   ├── InputData2026.xlsx
│   └── baseline/
│       ├── *_country.csv
│       └── *_forest.csv
├── outputs/
│   └── summary and selected reproducible output products
└── src/
    ├── load_libraries.R
    ├── files_manage.R
    ├── colors.R
    ├── corellation_filter.R
    ├── boruta.R
    ├── model_specification.R
    ├── metafit_ens.R
    ├── perturbation_audit.R
    └── post_processing/
        ├── all_predictions_check.R
        ├── compare_GCMs.R
        ├── correlation_matrix.R
        ├── extract_scenario_ensemble_means.R
        ├── extract_stacking_coefficients.R
        ├── plot_scenario_ensemble_means.R
        ├── select_model.R
        ├── sensitivity_test.R
        └── summarize.R
```

### Main files and folders

| Path | Purpose |
| --- | --- |
| `BFAI_model_MAIN.R` | Main modelling script used to run the full analysis. |
| `inputs/` | Input datasets used by the modelling workflow. |
| `inputs/baseline/` | Baseline climate-derived country and forest predictor tables. |
| `src/load_libraries.R` | Loads the R packages required by the workflow. |
| `src/corellation_filter.R` | Removes strongly correlated predictors before subsequent feature selection. |
| `src/boruta.R` | Performs Boruta feature selection. |
| `src/model_specification.R` | Defines the candidate regression algorithms used in the tidymodels workflow. |
| `src/metafit_ens.R` | Builds stacked ensemble models from tuned base learners. |
| `src/perturbation_audit.R` | Performs plausibility and robustness checks for scenario predictions. |
| `src/post_processing/` | Scripts for summaries, model comparison, scenario plots, stacking coefficients, and sensitivity checks. |
| `outputs/` | Output products. Large generated model objects are intentionally not versioned. |

---

## Data and predictors

The response variable is:

```text
BFAI = annual burnt forest area / national forest area × 1000
```

where BFAI is expressed in ha burnt per kha of forest.

The predictor set combines climate, drought, fuel-moisture, fire-weather, forest-composition, and country-level information. The manuscript describes 18 base climate and land-surface variables aggregated over full-year and seasonal windows, producing 90 climate-related predictors before feature selection. Together with forest composition and country, this gives 94 initial candidate predictors.

Main predictor families include:

| Predictor family | Examples | Interpretation |
| --- | --- | --- |
| Fire Weather Index | `FWI_DI`, `FWI_3+`, `FWI_4+`, `FWI_5+`, `FWI_6` | Fire-weather intensity and counts of days above danger thresholds. |
| Dead fuel moisture | `DFMC10H`, `DFMC10H_u10`, `DFMC10H_u8`, etc. | Mean or threshold-based 10-hour dead fuel moisture indicators. |
| Vapour pressure deficit | `VPD` | Atmospheric moisture demand derived from temperature and relative humidity. |
| Soil-water status | `AWR`, `AWP`, `AWD` | Soil drought and water-deficit indicators derived from SoilClim. |
| Forest composition | conifers, broadleaves, pines | Time-invariant country-level vegetation composition. |
| Country | categorical predictor | Captures country-specific factors such as management, reporting, land use, and institutional differences. |

Seasonal windows include:

- full year;
- March–October (`MAMJJASO`);
- April–September (`AMJJAS`);
- October–March (`ONDJFM`);
- November–February (`NDJF`).

---

## Modelling workflow

The modelling workflow follows the tidy modelling approach implemented in [`tidymodels`](https://www.tidymodels.org/) and described in [_Tidy Modeling with R_](https://www.tmwr.org/). In particular, the project uses recipes, workflows, model specifications, resampling, tuning, performance metrics, and stacked ensembles.

### 1. Data split

The annual observations are split into calibration and verification subsets using a repeated 3-year block structure:

- first and second years of each 3-year block: calibration;
- every third year: verification.

The response variable is log-transformed before modelling because BFAI is strongly right-skewed.

### 2. Feature selection

The workflow reduces the high-dimensional predictor set in three steps:

1. **Correlation filtering** removes multicollinear predictors.
2. **Boruta selection** identifies predictors that are consistently more informative than random shadow variables under cross-validation.
3. **Recursive feature elimination** ranks and retains the final predictor set using Random Forest variable importance.

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

### 3. Candidate models

The workflow evaluates 12 regression algorithms:

| Model family | Implemented models |
| --- | --- |
| Linear models | Generalized linear model, Elastic Net |
| Flexible regression | Multivariate Adaptive Regression Splines (MARS) |
| Support vector machines | Linear, polynomial, and radial kernels |
| Tree-based models | CART, bagged trees, Random Forest, Cubist |
| Boosting and Bayesian additive models | XGBoost, BART |
| Neural networks | Multilayer perceptron |

The model specifications are defined in `src/model_specification.R` using tidymodels-compatible engines such as `glmnet`, `nnet`, `earth`, `kernlab`, `rpart`, `ranger`, `xgboost`, `Cubist`, and `dbarts`.

### 4. Tuning and evaluation

Each candidate model is combined with one or more preprocessing strategies, including:

- no preprocessing;
- normalization of numerical predictors;
- polynomial terms and interactions.

Hyperparameters are tuned using:

- grid search;
- ANOVA racing via `finetune`.

Models are evaluated using 10-fold cross-validation on the calibration data and then independently assessed on the verification data. Main metrics include RMSE and coefficient of determination (`R²`).

### 5. Stacked ensemble

The final ensemble is built using stacked generalization through the `stacks` package. Predictions from tuned base learners are combined using penalized regression. The ensemble procedure is repeated with multiple random seeds to improve stability, and the final ensemble is selected based on calibration RMSE while preserving balanced calibration–verification performance.

---

## Climate-scenario projections

The project applies the trained ensemble model to climate-scenario predictors derived from CMIP6 global climate model simulations. The manuscript focuses on SSP2-4.5.

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
  "tidyverse", "tidymodels", "stacks", "finetune", "yardstick",
  "recipes", "workflows", "workflowsets", "parsnip", "tune",
  "Boruta", "caret", "ranger", "randomForest", "xgboost",
  "Cubist", "earth", "kernlab", "kknn", "nnet", "dbarts",
  "baguette", "rules", "embed", "vip", "readxl", "yaml",
  "future", "furrr", "doFuture", "parallel",
  "ggrepel", "ggtext", "cowplot", "patchwork", "ragg",
  "RColorBrewer", "gtable", "lme4"
))
```

Then load the project dependencies:

```r
source("src/load_libraries.R")
```

### 3. Run the main workflow

```r
source("BFAI_model_MAIN.R")
```

Depending on the number of candidate models, tuning settings, and available CPU cores, the full workflow can be computationally demanding. Large generated `.rds` model objects are not intended to be tracked in Git.

### 4. Post-processing

Selected post-processing scripts can be run after the model outputs are available:

```r
source("src/post_processing/summarize.R")
source("src/post_processing/extract_stacking_coefficients.R")
source("src/post_processing/plot_scenario_ensemble_means.R")
```

---

## Outputs

Typical generated outputs include:

- tuned model results;
- race and grid-search results;
- model stacks and ensemble objects;
- observed-versus-predicted plots;
- variable-importance figures;
- scenario projections by country and climate model;
- summary PDFs and tables for model configurations.

Large generated files such as model objects, tuning objects, and scenario `.rds` files should remain outside Git history. Use local storage, release assets, or external archival services for large derived outputs.

---

## Reproducibility notes

- The workflow depends on stochastic procedures including resampling, hyperparameter tuning, Random Forest models, and stacked ensemble construction. Use fixed seeds where exact reproducibility is required.
- Some outputs can differ slightly across operating systems or package versions, especially for parallel tuning and graphics devices.
- `ragg` is used for more stable cross-platform rendering of figures.
- The repository is designed for scientific reproducibility of the manuscript analysis, not as a general-purpose R package.

---

## Methodological basis

This workflow is based on three pillars:

1. **tidymodels** — a collection of R packages for modelling and machine learning using tidyverse principles: <https://www.tidymodels.org/>.
2. **Tidy Modeling with R** — the practical reference for model workflows, resampling, tuning, recipes, model comparison, and ensembles: <https://www.tmwr.org/>.
3. **Archived research workflow** — Zenodo DOI: <https://doi.org/10.5281/zenodo.17311553>.

---

## Citation

If you use this repository, please cite the associated manuscript and archived repository. Until the final article DOI is available, cite the repository as:

```text
Fischer, M. et al. BFAI: machine-learning assessment of European burnt forest area.
GitHub repository: https://github.com/MilanFischer/BFAI
Zenodo DOI: https://doi.org/10.5281/zenodo.17311553
```

Please also cite the methodological framework used by the analysis:

```text
Kuhn, M., & Wickham, H. tidymodels: a collection of packages for modeling and machine learning using tidyverse principles.
Kuhn, M., & Silge, J. Tidy Modeling with R. https://www.tmwr.org/
```

---

## License

This repository is distributed under the GNU General Public License v3.0 (GPL-3.0). See [`LICENSE`](LICENSE) for details.

---

## Contact

For questions about the code and reproducibility workflow, please open an issue in this repository or contact the repository maintainer.
