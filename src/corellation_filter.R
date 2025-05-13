corr_filter <- function(data, cutoff = 0.9, use_year = FALSE){
  #-------------------------------------------------------------------------------
  # Step 1: Remove Highly Correlated Features (keeping the one most correlated with log_BFA1000)
  cor_matrix <- cor(
    data |> select(-all_of(c(target, "Country", if (!use_year) "Year"))),
    use = "pairwise.complete.obs"
  )
  
  correlation_with_target <- cor(
    data |> select(-all_of(c(target, "Country", if (!use_year) "Year"))), 
    data |> pull(all_of(target)), 
    use = "pairwise.complete.obs"
  )
  
  high_corr <- findCorrelation(cor_matrix, cutoff = cutoff)
  high_corr_pairs <- combn(names(data)[-1], 2, simplify = FALSE)
  remove_vars <- numeric()
  
  # Identify highly correlated pairs and keep the one with stronger correlation with log_BFA1000
  for (i in 1:(ncol(cor_matrix) - 1)) {
    for (j in (i + 1):ncol(cor_matrix)) {
      if (!is.na(cor_matrix[i, j]) && abs(cor_matrix[i, j]) > cutoff) {
        var1 <- colnames(cor_matrix)[i]
        var2 <- colnames(cor_matrix)[j]
        if (abs(correlation_with_target[var1, 1]) >= abs(correlation_with_target[var2, 1])) {
          remove_vars <- c(remove_vars, var2)
        } else {
          remove_vars <- c(remove_vars, var1)
        }
      }
    }
  }
  
  keep_vars <- setdiff(names(data), unique(remove_vars))
  
  return(keep_vars)
}