import numpy as np
from scipy.special import beta as betafn
from rpy2 import robjects as robj
import numpy.random as npr
from scipy.stats import beta
import pymc3 as pm


# compute the KL divergence of beta ditributions with para1 and para2
# KL(beta(para1)||beta(para2))
def KL_dist_beta(para1, para2):
    a1, b1 = para1
    a2, b2 = para2
    sps = npr.beta(a1, b1, 10000)
    itm1 = (a1 - a2) * np.mean(np.log(sps))
    itm2 = (b1 - b2) * np.mean(np.log(1-sps))
    itm3 = - np.log(betafn(a1, b1)/betafn(a2, b2))
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
    
    
# Obtain the prior parameters for rMAP methods
def getMixBeta(Ds, w=0.1, mean=0.5):
    """
    Input:
        Ds: The historical datasets
        w: The robustness parameter
        mean: The mean parameter
    return:
        The prior parameters
    """
    robj.r.source("MAPBeta.R")
    Rmat = getBetaRDF(Ds)
    rmap = np.array(robj.r.getrMAPBeta(Ds=Rmat, w=w, mean=mean))
    return rmap


def getBetaRDF(Ds):
    """
    Input:
        Ds: The historical datasets
    return:
        The historical datasets in R matrix format
    """
    rs = []
    ns = []
    for Dh in Ds:
        rs.append(np.sum(Dh))
        ns.append(len(Dh))
    DF = rs + ns
    Rmat = robj.r.matrix(robj.FloatVector(DF), nrow=2, byrow=True)
    return Rmat


# Compute the posterior distributions for mixture priors
def MixPostBeta(paras, D):
    """
    Input:
        paras: The parameters for mixture prior, 3 by n
        D: The current dataset
    Return:
       postParas: The parameters for posterior distribution
    """
    nC = paras.shape[1]
    postParas = np.zeros((3, nC))
    postParas[1, :] = paras[1, :] + np.sum(D)
    postParas[2, :] = paras[2, :] + len(D) - np.sum(D)
    Cks = []
    for i in range(nC):
        postPara = postParas[1:, i]
        priorPara = paras[1:, i]
        Ck = betafn(postPara[0], postPara[1])/betafn(priorPara[0], priorPara[1])
        Cks.append(Ck)
    Cks = np.array(Cks)
    ws = Cks*paras[0, :]
    ws = ws/ws.sum()
    postParas[0, :] = ws
    return postParas


# Draw samples from mixture beta distrbution
def genMixBeta(num, paras):
    """
    Input:
        num:  The number of samples
        paras: The parameters for mixture prior, 3 by n
    Return:
        The samples
    """
    sps = np.zeros(num)
    nC = paras.shape[1]
    ws = paras[0, :]
    cumsumWs = np.cumsum(ws)
    uSps = np.random.uniform(size=num)
    cIdxs = np.sum(cumsumWs < uSps.reshape(-1, 1), axis=1)
    for k in range(nC):
        alp, bet = paras[1:, k]
        sps[cIdxs==k] = beta.rvs(a=alp, b=bet, size=np.sum(cIdxs==k))
    return sps
    
    
# Density of mixture beta distributoion
def denMixBeta(x, paras):
    """
    Input:
        x:  The value to be evaluated
        paras: The parameters for mixture prior, 3 by n
    """
    rv = 0
    nC = paras.shape[1]
    ws = paras[0, :]
    for k in range(nC):
        alp, bet = paras[1:, k]
        rv += ws[k]*beta.pdf(x, a=alp, b=bet)
    return rv




def popu_beta(alp, bt, cutoff):
    eqInt = ss.beta.ppf([cutoff, 1-cutoff], a=alp, b=bt)
    post_mean = alp/(alp+bt)
    sps = ss.beta.rvs(a=alp, b=bt, size=10000)
    HPDInt = pymcs.hpd(sps)
    return {"eq":eqInt, "post_mean":post_mean, "HPD": HPDInt}

def samp_beta(sps, cutoff):
    eqInt = np.quantile(sps, q=[cutoff, 1-cutoff])
    post_mean = np.mean(sps)
    HPDInt = pymcs.hpd(sps)
    return {"eq":eqInt, "post_mean":post_mean, "HPD": HPDInt}
