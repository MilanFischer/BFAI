create_next_output_dir <- function(base_dir = "./outputs") {
  dir_list <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  dir_list <- dir_list[grepl("out_\\d{3}", basename(dir_list))]
  
  if (length(dir_list) == 0) {
    out_path <- file.path(base_dir, "out_001")
  } else {
    nums <- as.integer(sub("out_(\\d{3})", "\\1", basename(dir_list)))
    next_dir <- sprintf("out_%03d", max(nums) + 1)
    out_path <- file.path(base_dir, next_dir)
  }
  
  dir.create(out_path)
  return(out_path)
}