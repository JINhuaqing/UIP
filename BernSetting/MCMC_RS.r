rm(list=ls())
setwd("C:/Users/Dell/Documents/ProjectCode/UIP")
library(magrittr)
library(dplyr)
source("utilities_bern.R")
# Run simulatio for UIP project 
# Bernulli distrubution

# true successful rate for Bernulli distribution
p0 <- 0.4
# the number of observation in current data
n <- 50
# current data
D <- rbinom(n, 1, p0)

# historical data parameters
ns <- c(40, 40)
ps <- c(0.5, 0.8)
Ds <- list()
for (i in 1:length(ns)){
  Ds[[i]] <- rbinom(ns[i], 1, ps[i])
}


post.sps1 <- gen.post.UIP.multi(10000, D, Ds)
post.sps.theta.1 <- post.sps1$sps


#IMH sample
accp.prob.f <- function(thetax, thetay, D){
   sumD <- sum(D) 
   nsumD <- length(D) - sum(D)
   logratio <- sumD*(log(thetay)-log(thetax)) + nsumD * (log(1-thetay) - log(1-thetax))
   if (logratio >= 0){
       return(1)
   }else{
       return(logratio)
   }
}
burnin <- 5000
interval <- 5
spsx <- list(sps=0.5, s.multis=list(matrix(c(10, 15), ncol=1)), s.pois=25)
seq.theta <- c()
seq.M <- c()
seq.ms <- list()
for (i in 1:10000){
   print(i)
   thetax <- spsx$sps
   #spsy <- gen.prior.UIP.KL(1, D, Ds, 25)
   spsy <- gen.prior.UIP.multi(1, Ds, 25)
   thetay <- spsy$sps
   ru <- runif(1)
   logru <- log(ru)
   logap <- accp.prob.f(thetax, thetay, D)
   if (logru <= logap){
       seq.theta <- c(seq.theta, spsy$sps)
       seq.M <- c(seq.M, spsy$s.pois)
       seq.ms[[i]] <- spsy$s.multis
       spsx <- spsy
   }else{
       seq.theta <- c(seq.theta, spsx$sps)
       seq.M <- c(seq.M, spsx$s.pois)
       seq.ms[[i]] <- spsx$s.multis
       spsx <- spsx
   }
}


post.sps.theta.2 <- seq.theta[burnin:10000]
post.sps.M.2 <- seq.M[burnin:10000]
keepidx <- seq(1, 5000, interval)
post.sps.theta.2 <- post.sps.theta.2[keepidx]
post.sps.M.2 <- post.sps.M.2[keepidx]
post.sps.theta.1 %>% mean
post.sps.theta.2 %>% mean
post.sps.theta.1 %>% sd
post.sps.theta.2 %>% sd

post.sps1$s.pois %>% mean
post.sps.M.2 %>% mean

post.sps.theta.1 %>% density() %>% plot(col="green", lwd=2)
post.sps.theta.2 %>% density() %>% lines(col="red", lwd=2)


