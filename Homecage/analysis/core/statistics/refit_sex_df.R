# Reusable statistical calculation; no command-line parsing.
refit_sex_df <- function(out) {
  library(nlme)
  library(emmeans)
  base <- 'analysis/output/Final'
  a <- readRDS(file.path(base,'FigS7AB/data/FigS7AB_sex_model.rds'))
  m <- a$model; d <- a$data
  stopifnot(is.matrix(m$apVar), all(eigen(m$apVar,symmetric=TRUE)$values>0))
  endpoints <- list(W4_W5_mean=c(0,.5,.5,0,0,0), W3_W6_AUC=c(1,2,2,1,0,0)/6)
  run_seed <- function(seed) {
   set.seed(seed); warnings <- character(); start <- proc.time()[3]
   ans <- tryCatch(withCallingHandlers({
    e <- emmeans(m,~condition*sex*week_f,data=d,mode='appx-satterthwaite',extra.iter=200)
    stopifnot(identical(attr(e@dffun,'mesg'),'appx-satterthwaite'))
    grid <- e@grid; C <- list(); meta <- list()
    for(ep in names(endpoints)) {
     means <- list()
     for(sx in c('m','f')) for(co in c('control','vpa')) {
      means[[paste(sx,co,sep='_')]] <- as.numeric(grid$condition==co & grid$sex==sx) *
        endpoints[[ep]][as.integer(as.character(grid$week_f))-2]
     }
     local <- list(
      condition=(means$m_vpa-means$m_control+means$f_vpa-means$f_control)/2,
      sex=(means$f_control-means$m_control+means$f_vpa-means$m_vpa)/2,
      condition_by_sex=(means$m_vpa-means$m_control)-(means$f_vpa-means$f_control))
     for(effect in names(local)) {
      nm <- paste(ep,effect,sep='__'); C[[nm]] <- local[[effect]]
      meta[[nm]] <- data.frame(endpoint=ep,effect=effect)
     }
    }
    z <- as.data.frame(summary(contrast(e,method=C,adjust='none'),infer=c(TRUE,TRUE)))
    z <- cbind(do.call(rbind,meta),z)
    z$numerator_df <- 1; z$F <- z$t.ratio^2; z$q_BH2 <- NA_real_; z$seed <- seed
    for(effect in unique(z$effect)) {
     ii <- z$effect==effect; z$q_BH2[ii] <- p.adjust(z$p.value[ii],method='BH')
    }
    L <- do.call(rbind,lapply(C,function(w)drop(w%*%e@linfct)))
    stopifnot(all(is.finite(z$df)),all(z$df>0),
     max(abs(z$p.value-pf(z$F,1,z$df,lower.tail=FALSE)))<1e-10,
     max(abs(z$estimate-drop(L%*%fixef(m))))<1e-8,
     max(abs(z$SE-sqrt(diag(L%*%vcov(m)%*%t(L)))))<1e-8)
    list(results=z,emmeans_grid=e,contrast_matrix=L)
   },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
   error=function(e)list(error=conditionMessage(e)))
   ans$warnings <- warnings; ans$seed <- seed; ans$seconds <- proc.time()[3]-start; ans
  }
  result <- run_seed(20260914)
  if(!is.null(result$error))stop(result$error)
  write.csv(result$results,file.path(out,'factorial_tests.csv'),row.names=FALSE)
  saveRDS(list(seed=result$seed,contrast_matrix=result$contrast_matrix,dfargs=result$emmeans_grid@dfargs),file.path(out,'inference_parameters.rds'))

}
