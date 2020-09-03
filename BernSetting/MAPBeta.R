library(RBesT)

getrMAPBeta <- function(Ds, w, mean){
    nD <- dim(Ds)[2] 
    stds <- 1:nD
    dataDF <- data.frame(study=stds, r=Ds[1, ], n=Ds[2, ])
    map_mcmc <- gMAP(cbind(r, n-r)~1|study, family=binomial, data=dataDF, tau.prior=1)
    map <- automixfit(map_mcmc)
    rmap <- robustify(map, weight=w, mean=mean)
    return(rmap)
}

