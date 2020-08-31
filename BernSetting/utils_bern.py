import numpy as np
import numpy.random as npr
from scipy.stats import norm, invgamma 
import pymc3 as pm
from scipy.special import beta as Beta


# compute the KL divergence of beta ditributions with para1 and para2
# KL(beta(para1)||beta(para2))
def KL_dist_beta(para1, para2):
    a1, b1 = para1
    a2, b2 = para2
    sps = npr.beta(a1, b1, 10000)
    itm1 = (a1 - a2) * np.mean(np.log(sps))
    itm2 = (b1 - b2) * np.mean(np.log(1-sps))
    itm3 = - np.log(Beta(a1, b1)/Beta(a2, b2))
    return itm1 + itm2 + itm3 

# compute the Jensen-Shannon Divergence
def JS_dist_beta(para1, para2):
    KL1 = KL_dist_beta(para1, para2)
    KL2 = KL_dist_beta(para2, para1)
    return 0.5* (KL1 + KL2)





# Obtain the weight parameters for JS method
def getUIPJSwsBern(D, Ds):
    n = len(D)
    ns = np.array([len(Dh) for Dh in Ds])
    parasc = [1+np.sum(D), 1+n-np.sum(D)]
    mInvs = []
    for nh, Dh in zip(ns, Ds):
        if nh <= n:
            parash = [1+np.sum(Dh), 1+nh-np.sum(Dh)]
        else:
            thetahat = np.mean(Dh)
            parash = [1+n*thetahat, 1+n*(1-thetahat)]
        mInvs.append(JS_dist_beta(parasc, parash))
    ms = 1/(np.array(mInvs)+1e-10)
    ws = ms/ms.sum()
    return ws


# Obtain the pymc context for UIP-D method
def getUIPDBern(D, Ds, upM=None):
    model = pm.Model()
    n = len(D)
    ns = np.array([len(Dh) for Dh in Ds])
    nD = len(Ds)
    if upM is None:
        upM = np.sum([len(Dh) for Dh in Ds])
    Means = [np.mean(Dh) for Dh in Ds]
    DirAlps = ns/n
    DirAlps[DirAlps>=1] = 1
    with model:
        pis = pm.Dirichlet("pis", DirAlps)
        M = pm.Uniform("M", lower=0, upper=upM)
        mean = 0
        varInv = 0  
        for i in range(nD):
            mean += pis[i]*Means[i]
            varInv += M*pis[i]/Means[i]/(1-Means[i])
        alp = mean * (mean*(1-mean)*varInv - 1)
        bet = (1-mean) * (mean*(1-mean)*varInv - 1)
        thetah = pm.Beta("thetah", alpha=alp, beta=bet)
        Yobs = pm.Bernoulli("Yobs", p=thetah, observed=D)
    return model 


# Obtain the pymc context for UIP-JS method
def getUIPJSBern(D, Ds, upM=None):
    model = pm.Model()
    n = len(D)
    ns = np.array([len(Dh) for Dh in Ds])
    nD = len(Ds)
    if upM is None:
        upM = np.sum([len(Dh) for Dh in Ds])
    Means = [np.mean(Dh) for Dh in Ds]
    pis = getUIPJSwsBern(D, Ds)
    with model:
        M = pm.Uniform("M", lower=0, upper=upM)
        mean = 0
        varInv = 0
        for i in range(nD):
            mean += pis[i]*Means[i]
            varInv += M*pis[i]/Means[i]/(1-Means[i])
        alp = mean * (mean*(1-mean)*varInv - 1)
        bet = (1-mean) * (mean*(1-mean)*varInv - 1)
        thetah = pm.Beta("thetah", alpha=alp, beta=bet)
        Yobs = pm.Bernoulli("Yobs", p=thetah, observed=D)
    return model 
        
            
# Obtain the pymc context for NPP method
def getNPPBern(D, Ds):
    model = pm.Model()
    n = len(D)
    nD = len(Ds)
    with model:
        alp = 1
        bet = 1
        gammas = pm.Uniform("gammas", lower=0, upper=1, shape=nD)
        for k in range(nD):
            alp += gammas[k]*np.sum(Ds[k])
            bet += gammas[k]*len(Ds[k]) - gammas[k]*np.sum(Ds[k])
        thetah = pm.Beta("thetah", alpha=alp, beta=bet)
        Yobs = pm.Bernoulli("Yobs", p=thetah, observed=D)
    return model
    