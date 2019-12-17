# the useful functions for UIP under Bernulli distribution
library(magrittr)
# for truncated poisson
library(extraDistr)

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
  Mss <- lapply(s.pois, function(s.poi){ms*s.poi/sum(ms)})
  sps <- lapply(Mss, function(Ms){paras <- cond.prior(Ds, Ms); rbeta(1, shape1=paras[1], shape2=paras[2])})
  #sps <- lapply(Mss, function(Ms){paras <- cond.prior(Ds, Ms); x = rbeta(1, shape1=paras[1], shape2=paras[2]); if(is.na(x))print(c(Ms, paras, x)); x})
  sps <- do.call(c, sps)
  list(sps=sps, s.pois=s.pois)
}

# genrate sample from posteior given D with UPDKL
gen.post.UIP.KL <- function(N, D, Ds, fct=0.5, Maxiter=50, Ns=10000){
  MLE <- mean(D)
  n <- length(D)
  Dsum <- sum(D)
  nDsum <- n - Dsum
  
  logden <- Dsum*log(MLE) + nDsum*log(1-MLE) 
  den <- exp(logden)
  sps.full <- c()
  s.pois.full <- c()
  flag <- 1
  
  while (length(sps.full) <= N){
    all <- gen.prior.UIP.KL(Ns, D, Ds, n*fct)
    sps <- all$sps
    lognums <- Dsum*log(sps) + nDsum*log(1-sps)
    nums <- exp(lognums)
    
    unifs <- runif(Ns)
    vs <- nums/den
    keepidx <- vs >= unifs
    sps.full <- c(sps.full, sps[keepidx])
    s.pois.full <- c(s.pois.full, all$s.pois[keepidx])
    flag <- flag + 1
    #print(c(flag, length(sps.full)))
    if (flag > Maxiter){
        break()
    }
      
  }
  list(sps=sps.full, s.pois=s.pois.full)
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
  #if (alpha < 0) print(c(av, bv))
  return(c(alpha, beta))
}



# generate sample from prior 
gen.prior.UIP.multi <- function(N, Ds, lam){
  numDs <- length(Ds)
  s.pois <- rtpois(N, lambda=lam, a=0, b=2*lam)
  s.multis <- lapply(s.pois, function(x)rmultinom(1, size=x, rep(1, numDs)/numDs))
  sps <- lapply(s.multis, function(Ms){paras <- cond.prior(Ds, Ms); rbeta(1, shape1=paras[1], shape2=paras[2])})
  sps <- do.call(c, sps)
  list(sps=sps, s.pois=s.pois, s.multis=s.multis)
}

# genrate sample from posteior given D
gen.post.UIP.multi <- function(N, D, Ds, fct=0.5, Maxiter=50, Ns=10000){
  MLE <- mean(D)
  n <- length(D)
  Dsum <- sum(D)
  nDsum <- n - Dsum
  
  logden <- Dsum*log(MLE) + nDsum*log(1-MLE) 
  den <- exp(logden)
  sps.full <- c()
  s.pois.full <- c()
  s.multis.full <- list()
  flag <- 1
  
  while (length(sps.full) <= N){
    all <- gen.prior.UIP.multi(Ns, Ds, n*fct)
    sps <- all$sps
    lognums <- Dsum*log(sps) + nDsum*log(1-sps)
    nums <- exp(lognums)
    unifs <- runif(Ns)
    vs <- nums/den # +1 for debug
    keepidx <- vs >= unifs
    
    sps.full <- c(sps.full, sps[keepidx])
    s.pois.full <- c(s.pois.full, all$s.pois[keepidx])
    s.multis.full <- c(s.multis.full, all$s.multis[keepidx])
    flag <- flag + 1
    #print(c(flag, length(sps.full)))
    if (flag > Maxiter){
        break()
    }
  }
  list(sps=sps.full, s.pois=s.pois.full, s.multis=s.multis.full)
}
