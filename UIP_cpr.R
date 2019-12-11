# compare UIP methods with Jeffrey's prior, uniform prior and full borrowing 

library(magrittr)
# To use the hpd function
library(TeachingDemos) 
# for truncated poisson
library(extraDistr)
library(dplyr)
library(parallel)
## JPP priors funtions
gen.conpostp.jpp <- function(D, Ds, gammas){
  alp <- sum(D) + sum(sapply(1:length(Ds), function(i){gammas[i]*sum(Ds[[i]])}))
  bt <- length(D) - sum(D) + sum(sapply(1:length(Ds), function(i){gammas[i]*(length(Ds[[i]])-sum(Ds[[i]]))}))
  rbeta(1, shape1=alp, shape2=bt)
}

gen.conpostga.jpp <- function(n, p, Dh){
  a <- sum(Dh)
  b <- length(Dh) - sum(Dh)
  unisps <- runif(n)
  den <- b*log(1-p) + a*log(p)
  num <- log(1+((1-p)^b*p^a-1)*unisps)
  num/den
}

gen.post.jpp <- function(N, D, Ds, burnin=5000){
  flag <- 0
  sps <- vector()
  gammass <- list()
  p <- 0.1
  for (i in 1:N){
    gammas <- sapply(Ds, function(Dh){gen.conpostga.jpp(1, p, Dh)})
    p <- gen.conpostp.jpp(D, Ds, gammas)
    if (i > burnin){
      sps <- c(sps, p)
      gammass[[i-burnin]] <- gammas
    }
  }
  list(sps=sps, gam=gammass)
}

# UIP-KL prior functions
# compute the KL divergence of beta ditributions with para1 and para2
# KL(beta(para1)||beta(para2))
KL.dist.beta <- function(para1, para2){
  a1 <- para1[1]
  b1 <- para1[2]
  a2 <- para2[1]
  b2 <- para2[2]
  sps <- rbeta(10000, shape1=a1, shape2=b1)
  itm1 <- (a1-a2)*mean(log(sps)) 
  itm2 <- (b1-b2)*mean(log(1-sps)) 
  itm3 <- - log(beta(a1, b1)/beta(a2, b2))
  itm1 + itm2 + itm3
}

# generate prior from UIP-KL prior
gen.prior.UIP.KL <- function(N, D, Ds, lam){
  paras <- lapply(Ds, function(Dh)c(1+sum(Dh), 1+length(Dh)-sum(Dh)))
  para0 <- c(1+sum(D), 1+length(D)-sum(D))
  ms <- lapply(paras, function(para){KL.dist.beta(para, para0)}) %>% do.call(c, args=.)
  ms <- 1/ms
  s.pois <- rtpois(N, lambda=lam, a=0, b=2*lam)
  Mss <- lapply(s.pois, function(s.poi){ms*s.poi})
  sps <- lapply(Mss, function(Ms){paras <- cond.prior(Ds, Ms); rbeta(1, shape1=paras[1], shape2=paras[2])})
  sps <- do.call(c, sps)
  sps
}

# genrate sample from posteior given D with UPDKL
gen.post.UIP.KL <- function(N, D, Ds, fct=0.5){
  MLE <- mean(D)
  n <- length(D)
  Dsum <- sum(D)
  nDsum <- n - Dsum
  
  logden <- Dsum*log(MLE) + nDsum*log(1-MLE) 
  den <- exp(logden)
  
  sps <- gen.prior.UIP.KL(N, D, Ds, n*fct)
  lognums <- Dsum*log(sps) + nDsum*log(1-sps)
  nums <- exp(lognums)
  
  unifs <- runif(N)
  vs <- nums/den
  sps[vs>=unifs]
}

# UIP-M prior functions
# compute corresponding (alpha, beta) given M1, M2, M3
cond.prior <- function(Ds, Ms){
  M <- sum(Ms)
  MLEs <- lapply(Ds, function(Dh){mean(Dh)}) %>% do.call(c, args=.)
  av <- sum(Ms*MLEs)/M
  Is <- 1/(MLEs*(1-MLEs))
  bv <- 1/sum(Ms*Is)
  alpha <- av * (av*(1-av)/bv - 1)
  beta <- (1-av) * (av*(1-av)/bv - 1)
  return(c(alpha, beta))
}

# generate sample from prior 
gen.prior.UIP.multi <- function(N, Ds, lam){
  s.pois <- rtpois(N, lambda=lam, a=0, b=2*lam)
  s.multis <- lapply(s.pois, function(x)rmultinom(1, size=x, rep(1, 3)/3))
  sps <- lapply(s.multis, function(Ms){paras <- cond.prior(Ds, Ms); rbeta(1, shape1=paras[1], shape2=paras[2])})
  sps <- do.call(c, sps)
  sps
}

# genrate sample from posteior given D
gen.post.UIP.multi <- function(N, D, Ds, fct=0.5){
  MLE <- mean(D)
  n <- length(D)
  Dsum <- sum(D)
  nDsum <- n - Dsum
  
  logden <- Dsum*log(MLE) + nDsum*log(1-MLE) 
  den <- exp(logden)
  
  sps <- gen.prior.UIP.multi(N, Ds, n*fct)
  lognums <- Dsum*log(sps) + nDsum*log(1-sps)
  nums <- exp(lognums)
  
  unifs <- runif(N)
  vs <- nums/den
  sps[vs>=unifs]
}

set.seed(3)
# Run simulatio for UIP project 
# Bernulli distrubution

Bern.Simu <- function(jj, p0){
    # true successful rate for Bernulli distribution
    p0 <- p0 
    # the number of observation in current data
    n <- 50
    ns <- c(40, 40, 40)
    ps <- c(0.4, 0.5, 0.6)
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
    post.sps.UIPm <- gen.post.UIP.multi(50000, D, Ds, fct=0.5)
    low.UIPm <- quantile(post.sps.UIPm, alpha)
    up.UIPm <- quantile(post.sps.UIPm, 1-alpha)
    CI.eq.UIPm <- c(low.UIPm, up.UIPm)
    post.mean.UIPm <- mean(post.sps.UIPm)
    HPD.UIPm <- emp.hpd(post.sps.UIPm)
    iter.res$UIPm <- list(mean=post.mean.UIPm, CI.eq=CI.eq.UIPm, HPD=HPD.UIPm)
    
    
    ## UIP KL
    post.sps.UIPkl <- gen.post.UIP.KL(50000, D, Ds, fct=0.5)
    low.UIPkl <- quantile(post.sps.UIPkl, alpha)
    up.UIPkl <- quantile(post.sps.UIPkl, 1-alpha)
    CI.eq.UIPkl <- c(low.UIPkl, up.UIPkl)
    post.mean.UIPkl <- mean(post.sps.UIPkl)
    HPD.UIPkl <- emp.hpd(post.sps.UIPkl)
    iter.res$UIPkl <- list(mean=post.mean.UIPkl, CI.eq=CI.eq.UIPkl, HPD=HPD.UIPkl)

    iter.res
}

p0 <- 0.3
Num <- 1000

results <- mclapply(1:Num, Bern.Simu, p0=p0, mc.cores=5)
save.name <- paste0("Bern", p0*100, "_Simi_", Num, ".RData")
save(results, file=save.name)
