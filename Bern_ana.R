library(magrittr)
rm(list=ls())
#setwd("C:/Users/Dell/Google Drive/multi-computers_folder/projects/UIP")



clean.res <- function(results){
    new.results <- list()
    flag <- 1
    i <- 1
    for (result in results){
        if (length(result) > 1){
            new.results[[flag]] <- result
            flag <- flag + 1
        #}else{
        #    print(result)
            #print(i)
        }
        i <- i+1
    }
    new.results
}

In.interval <- function(p0, inv){
    if ((p0>=inv[1]) & (p0<=inv[2])){
        return(1)
    }else{
        return(0)
    }
}

test.res.f <- function(result, p0){
    len <- length(result)
    res <- sapply(result[2:len], function(x){In.interval(p0, x$HPD)})
    res    
}

load("Bern50_Simi_1000.RData")
results <- clean.res(results)
te <- sapply(results, function(res)res$UIPm$Ms %>% length)
print(results %>% length)
#asf
p0 <- 0.5

ress <- lapply(results, function(result)test.res.f(result, p0=p0))
ress <- do.call(rbind, ress)
sizes <- 1 - colMeans(ress)
sizes

fs <- list.files(pattern="*.RData")
p0 <- 0.5
flag <- 1
powerss <- list()
for (fil in fs){
    load(fil)
    results <- clean.res(results)
    print(length(results))
    ress <- lapply(results, function(result)test.res.f(result, p0=p0))
    ress <- do.call(rbind, ress)
    powers <- 1 - colMeans(ress)
    powerss[[flag]] <- powers
    flag <- flag + 1
}

powerss <- do.call(rbind, powerss)
av.powers <- colMeans(powerss)
av.powers




load("Bern50_Simi_1000.RData")
full <- lapply(results, function(res){res$full$mean}) %>% 
    do.call(rbind, args=.)
non <- lapply(results, function(res){res$non$mean}) %>%
    do.call(rbind, args=.)
UIPkl <- lapply(results, function(res){res$UIPkl$mean}) %>%
    do.call(rbind, args=.)
