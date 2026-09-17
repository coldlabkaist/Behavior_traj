# Reusable statistical calculation; no command-line parsing.
verify_frozen_inference <- function(out) {
  library(nlme)
  library(clubSandwich)
  base <- 'analysis/output/Final'
  b <- file.path(base,'Fig4EFG')
  a <- readRDS(file.path(b,'data/Fig4_EFG_model_fit.rds'))
  m <- a$model; d <- a$data
  L <- as.matrix(read.csv(file.path(b,'data/Fig4_EFG_contrast_weights.csv'),row.names=1,check.names=FALSE))
  expected <- do.call(rbind,lapply(c('E','F','G'),function(p)read.csv(file.path(b,paste0('stat/Fig4_',p,'_statistics.csv')))))
  L <- L[expected$comparison,,drop=FALSE]
  v <- vcovCR(m,cluster=d$dam,type='CR1')
  z <- as.data.frame(linear_contrast(m,v,contrasts=L,test='Satterthwaite',p_values=TRUE))
  z$comparison <- rownames(L); z$estimate <- z$Est; z$t <- z$Est/z$SE
  one <- grepl('^(control|vpa)_W',z$comparison)
  z$p <- ifelse(one,pt(z$t,z$df,lower.tail=FALSE),z$p_val); z$q <- NA_real_
  ii <- z$comparison %in% c('E_W4','E_W5')
  z$q[ii] <- p.adjust(z$p[ii],'BH'); z$q[one] <- p.adjust(z$p[one],'BH')
  z$CI_L <- z$estimate-qt(.975,z$df)*z$SE
  z$CI_U <- z$estimate+qt(.975,z$df)*z$SE
  cols <- c('estimate','SE','t','df','p','q','CI_L','CI_U')
  errors <- data.frame(panel='Fig4EFG',field=cols,error=vapply(cols,function(k)max(abs(z[[k]]-expected[[k]]),na.rm=TRUE),numeric(1)))
  stopifnot(all(errors$error<1e-10))
  write.csv(z[,c('comparison',cols)],file.path(out,'Fig4EFG_inference.csv'),row.names=FALSE)

  b <- file.path(base,'FigS7AB')
  a <- readRDS(file.path(b,'data/FigS7AB_sex_model.rds'))
  compact <- readRDS(file.path(b,'data/FigS7AB_inference_parameters.rds'))
  expected <- read.csv(file.path(b,'stat/FigS7AB_factorial_tests.csv'))
  m <- a$model; L <- compact$contrast_matrix
  stopifnot(identical(rownames(L),as.character(expected$contrast)),compact$seed==20260914)
  df_from_saved <- function(k,a) {
   variance <- drop(t(k)%*%a$V%*%k)
   gradient <- vapply(a$G,function(G)drop(t(k)%*%G%*%k),numeric(1))
   2*variance^2/drop(t(gradient)%*%a$A%*%gradient)
  }
  estimate <- drop(L%*%fixef(m)); SE <- sqrt(diag(L%*%vcov(m)%*%t(L)))
  df <- apply(L,1,df_from_saved,a=compact$dfargs)
  t <- estimate/SE; F <- t^2; p <- pf(F,1,df,lower.tail=FALSE); q <- numeric(length(p))
  for(effect in unique(expected$effect)) {ii<-expected$effect==effect;q[ii]<-p.adjust(p[ii],'BH')}
  z <- data.frame(estimate=estimate,SE=SE,df=df,t.ratio=t,F=F,p.value=p,q_BH2=q,
   lower.CL=estimate-qt(.975,df)*SE,upper.CL=estimate+qt(.975,df)*SE)
  e <- data.frame(panel='FigS7AB',field=names(z),error=vapply(names(z),function(k)max(abs(z[[k]]-expected[[k]])),numeric(1)))
  stopifnot(all(e$error<1e-10))
  shared <- read.csv(file.path(base,'Fig4EFG/data/Fig4_EFG_model_observations.csv'))
  key <- function(x)paste(x$condition,x$cage_id,x$week,sep='|')
  ix <- match(key(a$data),key(shared))
  stopifnot(length(ix)==130,!anyNA(ix),!anyDuplicated(ix),max(abs(a$data$developmental_score-shared$developmental_score[ix]))<1e-12)
  write.csv(cbind(expected[,c('endpoint','effect','contrast')],z),file.path(out,'FigS7_inference.csv'),row.names=FALSE)
  write.csv(rbind(errors,e),file.path(out,'R_inference_errors.csv'),row.names=FALSE)
  cat('Frozen BOI and sex inference verified from Final.\n')

}
