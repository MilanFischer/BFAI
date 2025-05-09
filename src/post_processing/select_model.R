# Define the range of file indices
file_indices <- 1:31

# Construct file paths
base_path <- "d:/Git/GitHub/GitHubRepositories/BFAI/outputs"
file_paths <- sprintf("%s/out_%03d/error_stat.csv", base_path, file_indices)

# Read and process each file
ensemble_metrics <- map_dfr(file_paths, function(path) {
  read_csv(path, show_col_types = FALSE) |> 
    filter(model == "ensemble") |> 
    mutate(file = path)
})

# Find the file with the lowest verification RMSE for the ensemble
best_file <- ensemble_metrics |> 
  slice_min(order_by = verification, n = 10)

# View the best file
print(best_file)
