log_message <- function(...) {
  message(
    format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "),
    paste0(...)
  )
}