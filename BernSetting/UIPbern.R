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

# true successful rate for Bernulli distribution
p0 <- 0.3
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


## beta(1, 1) prior, no historical information
alpha.non <- 1+sum(D)
beta.non <- 1+n-sum(D)
low.non <- qbeta(alpha, shape1=alpha.non, shape2=beta.non)
up.non <- qbeta(1-alpha, shape1=alpha.non, shape2=beta.non)
CI.eq.non <- c(low.non, up.non)
post.mean.non <- alpha.non/(alpha.non+beta.non); post.mean.non
HPD.non <- hpd(qbeta, shape1=alpha.non, shape2=beta.non);HPD.non

## Jeffrey's prior
alpha.jef <- 0.5 + sum(D)
beta.jef <- 0.5 + n - sum(D)
low.jef <- qbeta(alpha, shape1=alpha.jef, shape2=beta.jef)
up.jef <- qbeta(1-alpha, shape1=alpha.jef, shape2=beta.jef)
CI.eq.jef <- c(low.jef, up.jef)
post.mean.jef <- alpha.jef/(alpha.jef+beta.jef); post.mean.jef
HPD.jef <- hpd(qbeta, shape1=alpha.jef, shape2=beta.jef);HPD.jef

## full borrowing
Dhfull <- do.call(c, Ds)
Dfull  <- c(D, Dhfull)
alpha.full <- 0.5 + sum(Dfull)
beta.full <- 0.5 + length(Dfull) - sum(Dfull)
low.full <- qbeta(alpha, shape1=alpha.full, shape2=beta.full)
up.full <- qbeta(1-alpha, shape1=alpha.full, shape2=beta.full)
CI.eq.full <- c(low.full, up.full)
post.mean.full <- alpha.full/(alpha.full+beta.full); post.mean.full
HPD.full <- hpd(qbeta, shape1=alpha.full, shape2=beta.full);HPD.full

## power prior

post.sps.jpp <- gen.post.jpp(10000, D, Ds)
post.sps.gammas.jpp <- post.sps.jpp$gam
post.sps.jpp <- post.sps.jpp$sps
#post.sps.jpp <- sapply(1:10000, function(i)gen.conpostp.jpp(D, Ds, c(0.5, 0.5, 0.5)))
low.jpp <- quantile(post.sps.jpp, alpha)
up.jpp <- quantile(post.sps.jpp, 1-alpha)
CI.eq.jpp <- c(low.jpp, up.jpp)
post.mean.jpp <- mean(post.sps.jpp); post.mean.jpp
HPD.jpp <- emp.hpd(post.sps.jpp); HPD.jpp

## UIP-multi

post.sps.UIPm.all <- gen.post.UIP.multi(10000, D, Ds, fct=0.5)
post.sps.UIPm <- post.sps.UIPm.all$sps
low.UIPm <- quantile(post.sps.UIPm, alpha)
up.UIPm <- quantile(post.sps.UIPm, 1-alpha)
CI.eq.UIPm <- c(low.UIPm, up.UIPm)
post.mean.UIPm <- mean(post.sps.UIPm); post.mean.UIPm
HPD.UIPm <- emp.hpd(post.sps.UIPm); HPD.UIPm


## UIP KL

post.sps.UIPkl.all <- gen.post.UIP.KL(100, D, Ds, fct=0.5)
post.sps.UIPkl <- post.sps.UIPkl.all$sps
low.UIPkl <- quantile(post.sps.UIPkl, alpha)
up.UIPkl <- quantile(post.sps.UIPkl, 1-alpha)
CI.eq.UIPkl <- c(low.UIPkl, up.UIPkl)
post.mean.UIPkl <- mean(post.sps.UIPkl); post.mean.UIPkl
HPD.UIPkl <- emp.hpd(post.sps.UIPkl); HPD.UIPkl


res <- rbind(c(post.mean.non, HPD.non),
             c(post.mean.UIPm, HPD.UIPm),
             c(post.mean.UIPkl, HPD.UIPkl)) %>% as.data.frame()
names(res) <- c("postmean", "HPDl", "HPDu")
res <- mutate(res, HPDdiff = HPDu-HPDl, absdiff=abs(p0-postmean))
row.names(res)<- c("non", "UIPm", "UIPkl")
res[, c(1, 5, 2, 3, 4)]


#draw the plot of the results
colss <- rainbow(6)
plot(CI.eq.non, c(1, 1), type="l", col=colss[1], lwd=2, xlim=c(0, 1), ylim=c(0, 6), yaxt='n', ylab="", xlab="")
points(post.mean.non, 1, pch=19, cex=1)
lines(CI.eq.jef, c(2, 2), type="l", col=colss[2], lwd=2)
points(post.mean.jef, 2, pch=19, cex=1)
lines(CI.eq.full, c(3, 3), type="l", col=colss[3], lwd=2)
points(post.mean.full, 3, pch=19, cex=1)
lines(CI.eq.UIPm, c(4, 4), type="l", col=colss[4], lwd=2)
points(post.mean.UIPm, 4, pch=19, cex=1)
lines(CI.eq.UIPkl, c(5, 5), type="l", col=colss[5], lwd=2)
points(post.mean.UIPkl, 5, pch=19, cex=1)
lines(CI.eq.jpp, c(6, 6), type="l", col=colss[6], lwd=2)
points(post.mean.jpp, 6, pch=19, cex=1)
abline(v=p0)
axis(2, 1:6, c("non-info", "jeffrey", "full-info", "UIPm", "UIP-KL", "JPP"), las=2)
axis(1, p0, expression(p[0]))
