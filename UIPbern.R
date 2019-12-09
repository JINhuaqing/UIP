library(magrittr)
# To use the hpd function
library(TeachingDemos) 
# for truncated poisson
library(extraDistr)
library(dplyr)
# Run simulatio for UIP project 
# Bernulli distrubution

# true successful rate for Bernulli distribution
p0 <- 0.3
# the number of observation in current data
n <- 40
# current data
D <- rbinom(n, 1, p0)

# historical data parameters
ns <- c(30, 30, 30)
ps <- c(0.2, 0.3, 0.6)
Ds <- list()
for (i in 1:length(ns)){
  Ds[[i]] <- rbinom(ns[i], 1, ps[i])
}


## beta(1, 1) prior, no historical information
alpha.non <- 1+sum(D)
beta.non <- 1+n-sum(D)
post.mean.non <- alpha.non/(alpha.non+beta.non); post.mean.non
HPD.non <- hpd(qbeta, shape1=alpha.non, shape2=beta.non);HPD.non


## power prior
## to be done

## UIP-multi
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

post.sps.UIPm <- gen.post.UIP.multi(30000, D, Ds, fct=0.5)
post.mean.UIPm <- mean(post.sps.UIPm); post.mean.UIPm
HPD.UIPm <- emp.hpd(post.sps.UIPm); HPD.UIPm


## UIP KL
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

post.sps.UIPkl <- gen.post.UIP.KL(30000, D, Ds, fct=0.5)
post.mean.UIPkl <- mean(post.sps.UIPkl); post.mean.UIPkl
HPD.UIPkl <- emp.hpd(post.sps.UIPkl); HPD.UIPkl


res <- rbind(c(post.mean.non, HPD.non),
             c(post.mean.UIPm, HPD.UIPm),
             c(post.mean.UIPkl, HPD.UIPkl)) %>% as.data.frame()
names(res) <- c("postmean", "HPDl", "HPDu")
res <- mutate(res, HPDdiff = HPDu-HPDl, absdiff=abs(p0-postmean))
row.names(res)<- c("non", "UIPm", "UIPkl")
res[, c(1, 5, 2, 3, 4)]
