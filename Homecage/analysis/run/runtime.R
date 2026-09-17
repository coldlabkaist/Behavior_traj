# Shared R command setup; package installation remains a separate environment task.
prepare_output <- function(default, work_only=TRUE) {
  if (dir.exists('.r-library')) .libPaths(c(normalizePath('.r-library'), .libPaths()))
  args <- commandArgs(TRUE)
  stopifnot(length(args) <= 1)
  out <- if(length(args)) args[1] else default
  root <- normalizePath('.', winslash='/', mustWork=TRUE)
  target <- normalizePath(out, winslash='/', mustWork=FALSE)
  if(!grepl('^([A-Za-z]:|/)', target)) target <- paste0(root,'/',target)
  stopifnot(!grepl('(^|/)\\.\\.(/|$)',target))
  if(work_only) stopifnot(startsWith(tolower(target),tolower(paste0(root,'/analysis/work/'))))
  protected_paths <- c(file.path(root,c('analysis/output','data','checkpoints')),
                       file.path(dirname(root),c('MovAl_benchmark','direct_interaction','behavior_assay','data')))
  for(protected in tolower(protected_paths)) {
    stopifnot(tolower(target) != protected, !startsWith(tolower(target),paste0(protected,'/')))
  }
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  out
}
