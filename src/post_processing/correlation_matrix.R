# Select numeric predictors only
numeric_predictors <- calibration_data |> 
  select(all_of(predictors)) |> 
  select(where(is.numeric))  # Exclude "Country" or any non-numeric

# Compute correlation matrix
cor_matrix <- cor(numeric_predictors, use = "pairwise.complete.obs", method = "pearson")

# View the correlation matrix
print(cor_matrix)

# Optional: Visualize it
library(corrplot)
# Define the output file
png("correlation_plot.png", width = 1400, height = 1300, res = 150)

# Generate the correlation plot
corrplot::corrplot(
  cor_matrix, 
  method = "color", 
  type = "upper", 
  tl.cex = 1.2,
  addCoef.col = "black",
  tl.col = "black"
)

# Close the graphics device
dev.off()