library(RBesT)

getrMAPNormal <- function(Ds, w, mean){
    nD <- dim(Ds)[2] 
    stds <- 1:nD
    sighat <- Ds[2, 1] * sqrt(Ds[3, 1])
    dataDF <- data.frame(study=stds, y=Ds[1, ], y.se=Ds[2, ], n=Ds[3, ])
    map_mcmc <- gMAP(cbind(y, y.se)~1|study, weights=n, 
                     data=dataDF, family=gaussian, tau.dist="HalfNormal",
                     tau.prior=cbind(0, sighat/2), beta.prior=cbind(0, sighat))
    map <- automixfit(map_mcmc)
    rmap <- robustify(map, weight=w, mean=mean)
    return(rmap)
}
