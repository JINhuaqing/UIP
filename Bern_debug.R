# TO debug
rm(list=ls())
library(magrittr)
# To use the hpd function
library(TeachingDemos) 
# for truncated poisson
library(extraDistr)
library(dplyr)
source("utilities_bern.R")
# Run simulatio for UIP project 
# Bernulli distrubution

set.seed(1)
for (iii in 1:1000){
print(iii)
# true successful rate for Bernulli distribution
p0 <- 0.4
# the number of observation in current data
n <- 50
# current data
D <- rbinom(n, 1, p0)

# historical data parameters
ns <- c(40, 40)
ps <- c(0.45, 0.85)
Ds <- list()
for (i in 1:length(ns)){
  Ds[[i]] <- rbinom(ns[i], 1, ps[i])
}

alpha <- 0.025
## UIP-multi

post.sps.UIPm.all <- gen.post.UIP.multi(10000, D, Ds, fct=0.5, Maxiter=50)
post.sps.UIPm <- post.sps.UIPm.all$sps
print(post.sps.UIPm[is.nan(post.sps.UIPm)])
low.UIPm <- quantile(post.sps.UIPm, alpha)
up.UIPm <- quantile(post.sps.UIPm, 1-alpha)
CI.eq.UIPm <- c(low.UIPm, up.UIPm)
post.mean.UIPm <- mean(post.sps.UIPm); post.mean.UIPm
HPD.UIPm <- emp.hpd(post.sps.UIPm); HPD.UIPm

}