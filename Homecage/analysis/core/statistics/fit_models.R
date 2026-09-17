# Reusable statistical calculation; no command-line parsing.
refit_models <- function(out) {
  library(nlme)
  base <- 'analysis/output/Final'
  a <- readRDS(file.path(base,'Fig4EFG/data/Fig4_EFG_model_fit.rds'))
  # Frozen model data preserves the original factor levels, row order and hierarchy.
  d <- a$data
  observed <- read.csv(file.path(base,'Fig4EFG/data/Fig4_EFG_model_observations.csv'))
  key <- function(x) paste(x$condition,x$cage_id,x$week,sep='|')
  ix <- match(key(d),key(observed))
  stopifnot(nrow(d)==130,!anyNA(ix),!anyDuplicated(ix),max(abs(d$developmental_score-observed$developmental_score[ix]))<1e-12)
  X <- model.matrix(~condition*week_f,d)
  ctrl <- lmeControl(maxIter=100,msMaxIter=250,msMaxEval=1500,niterEM=40,
   tolerance=1e-6,msTol=1e-7,returnObject=FALSE)
  fit_one <- function(dat,structure,serial,optimizer='nlminb') {
   ctl <- ctrl; ctl$opt <- optimizer
   weights <- if(structure=='multiplicative') varComb(varIdent(form=~1|condition),varIdent(form=~1|week_f)) else varIdent(form=~1|cell)
   lme(developmental_score~condition*week_f,data=dat,
       random=~1|dam/litter/cage,
       weights=weights,correlation=if(serial) corAR1(form=~week|dam/litter/cage) else NULL,
       method='REML',control=ctl,na.action=na.fail)
  }
  covariance <- function(m) {
   sdres <- m$sigma/as.numeric(varWeights(m$modelStruct$varStruct))
   rho <- if(is.null(m$modelStruct$corStruct)) 0 else as.numeric(coef(m$modelStruct$corStruct,unconstrained=FALSE))
   R <- outer(sdres,sdres)*rho^abs(outer(d$week,d$week,'-'))*outer(d$cage,d$cage,'==')
   taus <- sapply(m$modelStruct$reStruct,function(z)m$sigma^2*as.matrix(pdMatrix(z))[1,1])
   for(g in names(taus)) R <- R+taus[g]*outer(d[[g]],d[[g]],'==')
   list(V=R,sdres=sdres,rho=rho,taus=taus)
  }
  fits <- list(); specs <- expand.grid(structure=c('multiplicative','cell'),serial=c(FALSE,TRUE),stringsAsFactors=FALSE)
  rows <- list()
  for(i in seq_len(nrow(specs))) {
   spec <- specs[i,]; id <- paste(spec$structure,if(spec$serial)'AR1' else 'RI',sep='_')
   cat('Fitting',id,'\n');flush.console()
   tm <- system.time(m <- tryCatch(fit_one(d,spec$structure,spec$serial),error=function(e)e))
   if(inherits(m,'error')) { rows[[id]] <- data.frame(model=id,structure=spec$structure,serial=spec$serial,
    converged=FALSE,AIC=NA,k=NA,apVar_ok=FALSE,min_random_variance=NA,rho=NA,error=conditionMessage(m));next }
   cv <- covariance(m); ViX <- solve(cv$V,X)
   beta <- solve(crossprod(X,ViX),crossprod(ViX,d$developmental_score))
   stopifnot(max(abs(beta-fixef(m)))<1e-5)
   stopifnot(max(abs(solve(crossprod(X,ViX))-vcov(m)))<1e-5)
   fits[[id]] <- m
   rows[[id]] <- data.frame(model=id,structure=spec$structure,serial=spec$serial,
    converged=TRUE,AIC=AIC(m),k=attr(logLik(m),'df'),apVar_ok=is.matrix(m$apVar),
    min_random_variance=min(cv$taus),rho=cv$rho,error='')
   cat('Completed',id,'AIC',AIC(m),'random variances',cv$taus,'rho',cv$rho,'\n');flush.console()
  }
  comparison <- do.call(rbind,rows);rownames(comparison) <- NULL
  write.csv(comparison,file.path(out,'covariance_candidates.csv'),row.names=FALSE)
  # Same fixed means and REML throughout. Within delta AIC <=2 choose fewer parameters.
  ok <- subset(comparison,converged)
  pool <- subset(ok,AIC<=min(AIC)+2)
  pick <- pool[order(pool$k,pool$AIC),][1,]
  id <- pick$model; m <- fits[[id]]; cv <- covariance(m)
  cat('Selected covariance before gain tests:',id,'\n');flush.console()

  saveRDS(list(model=m,id=id,spec=pick,data=d,X=X,covariance=cv,comparison=comparison),file.path(out,'BOI_model_fit.rds'))
  errors <- data.frame(model='BOI',beta=max(abs(fixef(m)-fixef(a$model))),covariance=max(abs(vcov(m)-vcov(a$model))))

  # Sex factorial model, including the original numerical optimizer fallback.
  a <- readRDS(file.path(base,'FigS7AB/data/FigS7AB_sex_model.rds'))
  d <- a$data
  d$sex <- factor(d$sex,levels=c('m','f'))
  ids <- unique(d[c('condition','sex','cage_id','dam','litter')])
  X <- model.matrix(~condition*sex*week_f,d)
  stopifnot(qr(X)$rank==ncol(X),ncol(X)==24)
  ctl <- lmeControl(maxIter=100,msMaxIter=250,msMaxEval=1500,niterEM=40,
   tolerance=1e-6,msTol=1e-7,returnObject=FALSE)
  warnings_fit<-character()
  m<-withCallingHandlers(lme(developmental_score~condition*sex*week_f,data=d,
   random=~1|dam/litter/cage,
   weights=varComb(varIdent(form=~1|condition),varIdent(form=~1|week_f)),
   correlation=NULL,method='REML',control=ctl,na.action=na.fail),
   warning=function(w){warnings_fit<<-c(warnings_fit,conditionMessage(w));invokeRestart('muffleWarning')})
  # Self-contained update call for the emmeans numerical df approximation.
  m$call<-quote(nlme::lme(fixed=developmental_score~condition*sex*week_f,data=d,
   random=~1|dam/litter/cage,
   weights=nlme::varComb(nlme::varIdent(form=~1|condition),nlme::varIdent(form=~1|week_f)),
   correlation=NULL,method='REML',na.action=na.fail,
   control=nlme::lmeControl(maxIter=100,msMaxIter=250,msMaxEval=1500,niterEM=40,
   tolerance=1e-6,msTol=1e-7,returnObject=FALSE)))
  alt<-tryCatch(update(m,control=lmeControl(maxIter=100,msMaxIter=250,msMaxEval=1500,niterEM=40,
   tolerance=1e-6,msTol=1e-7,returnObject=FALSE,opt='optim')),error=function(e)e)
  primary_fit<-m
  optimizer_comparison<-if(inherits(alt,'error'))conditionMessage(alt) else c(
   loglik_diff=as.numeric(logLik(m)-logLik(alt)),
   max_beta_difference=max(abs(fixef(m)-fixef(alt))),max_vcov_difference=max(abs(vcov(m)-vcov(alt))))
  selected_optimizer<-'nlminb'
  # Numerical fallback only: same model, practically identical optimum, before any p values.
  if(!is.matrix(m$apVar) && !inherits(alt,'error') && is.matrix(alt$apVar) &&
   all(eigen(alt$apVar,symmetric=TRUE)$values>0) &&
   abs(optimizer_comparison['loglik_diff'])<1e-3 &&
   optimizer_comparison['max_beta_difference']<1e-4 && optimizer_comparison['max_vcov_difference']<1e-4) {
   m<-alt;selected_optimizer<-'optim'
   m$call$control<-quote(nlme::lmeControl(maxIter=100,msMaxIter=250,msMaxEval=1500,niterEM=40,
   tolerance=1e-6,msTol=1e-7,returnObject=FALSE,opt='optim'))
   cat('Using optim numerical fallback: same model, log-likelihood difference <.001, beta/vcov differences <.0001.\n')
  }
  sdres<-m$sigma/as.numeric(varWeights(m$modelStruct$varStruct))
  V<-diag(sdres^2)
  tau<-sapply(m$modelStruct$reStruct,function(z)m$sigma^2*as.matrix(pdMatrix(z))[1,1])
  for(g in names(tau))V<-V+tau[g]*outer(d[[g]],d[[g]],'==')
  ViX<-solve(V,X); independent_V<-solve(crossprod(X,ViX))
  independent_b<-drop(independent_V%*%crossprod(ViX,d$developmental_score))
  stopifnot(max(abs(independent_b-fixef(m)))<1e-5,
   max(abs(independent_V-vcov(m)))<1e-5)
  diagnostics<-list(fit_warnings=warnings_fit,selected_optimizer=selected_optimizer,
   primary_apVar=primary_fit$apVar,primary_random_variances=sapply(primary_fit$modelStruct$reStruct,function(z)primary_fit$sigma^2*as.matrix(pdMatrix(z))[1,1]),random_variances=tau,
   apVar_numeric=is.matrix(m$apVar),
   apVar_eigenvalues=if(is.matrix(m$apVar))eigen(m$apVar,symmetric=TRUE)$values else m$apVar,
   alternate_optimizer=optimizer_comparison,
   fixed_design_rank=qr(X)$rank,n_observations=nrow(d),n_cages=nrow(ids),
   max_abs_conditional_pearson_residual=max(abs(residuals(m,type='pearson'))))
  saveRDS(list(model=m,data=d,V=V,diagnostics=diagnostics),file.path(out,'sex_model.rds'))
  errors <- rbind(errors,data.frame(model='sex',beta=max(abs(fixef(m)-fixef(a$model))),covariance=max(abs(vcov(m)-vcov(a$model)))))
  write.csv(errors,file.path(out,'model_refit_comparison.csv'),row.names=FALSE)
  stopifnot(all(errors$beta<1e-7),all(errors$covariance<1e-7))
  print(errors)

}
