# compare UIP methods with Jeffrey's prior, uniform prior and full borrowing 

rm(list=ls())
library(magrittr)
# To use the hpd function
library(TeachingDemos) 
# for truncated poisson
library(extraDistr)
library(dplyr)
library(parallel)
source("utilities_bern.R")
set.seed(3)
# Run simulatio for UIP project 
# Bernulli distrubution

Bern.Simu <- function(jj, p0){
    # true successful rate for Bernulli distribution
    p0 <- p0 
    # the number of observation in current data
    n <- 50
    ns <- c(40, 40)
    ps <- c(0.45, 0.85)
    iter.res <- list()

    # current data
    D <- rbinom(n, 1, p0)
    
    # historical data parameters
    Ds <- list()
    for (i in 1:length(ns)){
      Ds[[i]] <- rbinom(ns[i], 1, ps[i])
    }
    
    alpha <- 0.025
    iter.res$data <- list(current=D, histD=Ds, p0=p0, ps=ps, n=n, ns=ns)
    print(c(jj, p0))
    
    
    ## beta(1, 1) prior, no historical information
    alpha.non <- 1+sum(D)
    beta.non <- 1+n-sum(D)
    low.non <- qbeta(alpha, shape1=alpha.non, shape2=beta.non)
    up.non <- qbeta(1-alpha, shape1=alpha.non, shape2=beta.non)
    CI.eq.non <- c(low.non, up.non)
    post.mean.non <- alpha.non/(alpha.non+beta.non)
    HPD.non <- hpd(qbeta, shape1=alpha.non, shape2=beta.non)
    iter.res$non <- list(mean=post.mean.non, CI.eq=CI.eq.non, HPD=HPD.non)
    
    ## Jeffrey's prior
    alpha.jef <- 0.5 + sum(D)
    beta.jef <- 0.5 + n - sum(D)
    low.jef <- qbeta(alpha, shape1=alpha.jef, shape2=beta.jef)
    up.jef <- qbeta(1-alpha, shape1=alpha.jef, shape2=beta.jef)
    CI.eq.jef <- c(low.jef, up.jef)
    post.mean.jef <- alpha.jef/(alpha.jef+beta.jef)
    HPD.jef <- hpd(qbeta, shape1=alpha.jef, shape2=beta.jef)
    iter.res$jef <- list(mean=post.mean.jef, CI.eq=CI.eq.jef, HPD=HPD.jef)
    
    ## full borrowing
    Dhfull <- do.call(c, Ds)
    Dfull  <- c(D, Dhfull)
    alpha.full <- 0.5 + sum(Dfull)
    beta.full <- 0.5 + length(Dfull) - sum(Dfull)
    low.full <- qbeta(alpha, shape1=alpha.full, shape2=beta.full)
    up.full <- qbeta(1-alpha, shape1=alpha.full, shape2=beta.full)
    CI.eq.full <- c(low.full, up.full)
    post.mean.full <- alpha.full/(alpha.full+beta.full)
    HPD.full <- hpd(qbeta, shape1=alpha.full, shape2=beta.full)
    iter.res$full <- list(mean=post.mean.full, CI.eq=CI.eq.full, HPD=HPD.full)
    
    ## power prior
    post.sps.jpp <- gen.post.jpp(10000, D, Ds)
    post.sps.gammas.jpp <- post.sps.jpp$gam
    post.sps.jpp <- post.sps.jpp$sps
    low.jpp <- quantile(post.sps.jpp, alpha)
    up.jpp <- quantile(post.sps.jpp, 1-alpha)
    CI.eq.jpp <- c(low.jpp, up.jpp)
    post.mean.jpp <- mean(post.sps.jpp)
    HPD.jpp <- emp.hpd(post.sps.jpp)
    iter.res$jpp <- list(mean=post.mean.jpp, CI.eq=CI.eq.jpp, HPD=HPD.jpp)
    
    ## UIP-multi
    post.sps.UIPm.all <- gen.post.UIP.multi(10000, D, Ds, fct=0.5)
    post.sps.UIPm <- post.sps.UIPm.all$sps
    low.UIPm <- quantile(post.sps.UIPm, alpha)
    up.UIPm <- quantile(post.sps.UIPm, 1-alpha)
    CI.eq.UIPm <- c(low.UIPm, up.UIPm)
    post.mean.UIPm <- mean(post.sps.UIPm)
    HPD.UIPm <- emp.hpd(post.sps.UIPm)
    iter.res$UIPm <- list(mean=post.mean.UIPm, CI.eq=CI.eq.UIPm, HPD=HPD.UIPm, Ms=post.sps.UIPm.all$s.multis, M=post.sps.UIPm.all$s.pois)
    
    
    ## UIP KL
    post.sps.UIPkl.all <- gen.post.UIP.KL(10000, D, Ds, fct=0.5)
    post.sps.UIPkl <- post.sps.UIPkl.all$sps
    low.UIPkl <- quantile(post.sps.UIPkl, alpha)
    up.UIPkl <- quantile(post.sps.UIPkl, 1-alpha)
    CI.eq.UIPkl <- c(low.UIPkl, up.UIPkl)
    post.mean.UIPkl <- mean(post.sps.UIPkl)
    HPD.UIPkl <- emp.hpd(post.sps.UIPkl)
    iter.res$UIPkl <- list(mean=post.mean.UIPkl, CI.eq=CI.eq.UIPkl, HPD=HPD.UIPkl, Ms=post.sps.UIPkl.all$s.pois)

    iter.res
}

p0 <- 0.3
Num <- 1000

results <- mclapply(1:Num, Bern.Simu, p0=p0, mc.cores=5)
save.name <- paste0("Bern", p0*100, "_Simi_", Num, ".RData")
save(results, file=save.name)
