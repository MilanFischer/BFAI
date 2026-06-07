library(tidyverse)
library(stacks)
library(broom)

# Load stack
models_stack <- readRDS(
  "./outputs/out_002/models_stack.rds"
)

# Blend
models_stack_blend <- models_stack |>
  blend_predictions(
    penalty = 0.01
  )

# Extract non-zero coefficients
coef_tbl <- tidy(models_stack_blend$coefs) |>
  filter(term != "(Intercept)") |>
  filter(estimate > 0) |>
  mutate(
    weight_percent = 100 * estimate / sum(estimate)
  ) |>
  arrange(desc(weight_percent))

coef_tbl