# Run from the project root: Rscript analysis/run/install_r_packages.R
# Install missing statistical dependencies; keep already available packages.
local_library <- '.r-library'
dir.create(local_library, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(normalizePath(local_library), .libPaths()))
required <- c('nlme', 'clubSandwich', 'sandwich', 'emmeans')
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
if(length(missing)) {
  install.packages(missing, lib=local_library, repos='https://cloud.r-project.org')
}
stopifnot(all(vapply(required, requireNamespace, logical(1), quietly=TRUE)))
for(name in required) cat(name, as.character(packageVersion(name)), '\n')
