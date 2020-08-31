import numpy as np
import numpy.random as npr
from scipy.stats import norm, invgamma 
import pymc3 as pm


def sigma2cond_theta(theta, D):
    n = len(D)
    shape = n/2 + 1/2
    scale = np.sum((D-theta)**2)/2
    return invgamma.rvs(a=shape, scale=scale, size=1)
    

def thetacond_sigma2(sigma2, D):
    n = len(D)
    loc = np.mean(D)
    scale = np.sqrt(sigma2/n) 
    return norm.rvs(loc=loc, scale=scale, size=1)


def gen_post_jef(N, D, burnin=5000, thin=5, diag=False):
    thetas = []
    sigma2s = []
    ctheta = 0
    for i in range(N):
        csigma2 = sigma2cond_theta(ctheta, D=D)
        ctheta = thetacond_sigma2(csigma2, D=D)
        thetas.append(ctheta)
        sigma2s.append(csigma2)
    thetas, sigma2s = np.array(thetas), np.array(sigma2s)
    if diag:
        return {"theta": thetas, "sigma2": sigma2s}
    else:
        return {"theta": thetas[burnin::thin], "sigma2": sigma2s[burnin::thin]}



def gen_post_full(N, D, Ds, burnin=5000, thin=5, diag=False):
    Dhall = np.concatenate(Ds)
    D = np.concatenate((D, Dhall))
    thetas = []
    sigma2s = []
    ctheta = 0
    for i in range(N):
        csigma2 = sigma2cond_theta(ctheta, D=D)
        ctheta = thetacond_sigma2(csigma2, D=D)
        thetas.append(ctheta)
        sigma2s.append(csigma2)
    thetas, sigma2s = np.array(thetas), np.array(sigma2s)
    if diag:
        return {"theta": thetas, "sigma2": sigma2s}
    else:
        return {"theta": thetas[burnin::thin], "sigma2": sigma2s[burnin::thin]}


def getUIPDNormal(D, Ds):
    nD = len(Ds)
    n = len(D)
    ns = np.array([len(Dh) for Dh in Ds])
    dalps = ns/n
    dalps[dalps>=1] = 1
    nsSum = np.sum([len(Dh) for Dh in Ds])
    Means = [np.mean(Dh) for Dh in Ds]
    Vars  = [np.var(Dh) for Dh in Ds]
    model = pm.Model()
    with model:
        pis = pm.Dirichlet("pis", dalps)
        sigma2 = pm.InverseGamma("sigma2", alpha=0.01, beta=0.01)
        M = pm.Uniform("M", lower=0, upper=nsSum)

        thetan = 0
        sigma2n_inv = 0
        for i in range(nD):
            thetan += pis[i] * Means[i]
            sigma2n_inv += pis[i] / Vars[i]
        sigma2n = 1/M/sigma2n_inv
        thetah = pm.Normal("thetah", mu=thetan, sigma=np.sqrt(sigma2n))

        Yobs = pm.Normal("Yobs", mu=thetah, sigma=np.sqrt(sigma2), observed=D)
    return model 

# Obtain the normal parameters for JS method
def obtainInitPost(Dh, nCur):
    sigma20 = 100
    nh = len(Dh)
    if nh <= nCur:
        sigma2hat = np.var(Dh)
        muInit = sum(Dh)/(nh + sigma2hat/sigma20)
        sigma2Init = 1 / (1/sigma20+nh/sigma2hat)
    else:
        muInits = []
        sigma2Inits = []
        for j in range(100):
            DhCur = np.random.choice(Dh, size=nCur, replace=False)
            sigma2hatCur = np.var(DhCur)
            muInitCur = sum(DhCur)/(nCur + sigma2hatCur/sigma20)
            sigma2InitCur = 1 / (1/sigma20+nCur/sigma2hatCur)
            muInits.append(muInitCur)
            sigma2Inits.append(sigma2InitCur)
        muInit = np.mean(muInits)
        sigma2Init = np.mean(sigma2Inits)
    return muInit, sigma2Init

def getUIPJSNormal(D, Ds, diag=False):
    def KLnorm(mu1, mu2, sigma1, sigma2):
        itm1 = np.log(sigma2/sigma1)
        itm2 = (sigma1**2 + (mu2-mu1)**2)/(2*sigma2**2) - 0.5
        return itm1 + itm2
    def JSnorm(mu1, mu2, sigma1, sigma2):
        KL1 = KLnorm(mu1, mu2, sigma1, sigma2)
        KL2 = KLnorm(mu2, mu1, sigma2, sigma1)
        return (KL1 + KL2)/2

    nD = len(Ds)
    n = len(D)
    nsSum = np.sum([len(Dh) for Dh in Ds])
    Means = [np.mean(Dh) for Dh in Ds]
    Vars  = [np.var(Dh) for Dh in Ds]
    parasc = obtainInitPost(D, n)
    invPis = []
    for idx, Dh in enumerate(Ds):
        parash = obtainInitPost(Dh, n)
        invPis.append(JSnorm(parasc[0], parash[0], parasc[1]**0.5, parash[1]**0.5))
   
    Pis = 1/(np.array(invPis) + 1e-10)
    pis = np.array(Pis)/np.sum(Pis)

    UIPJS = pm.Model()
    with UIPJS:
        sigma2 = pm.InverseGamma("sigma2", alpha=0.01, beta=0.01)
        M = pm.Uniform("M", lower=0, upper=nsSum)
        thetan = 0
        sigma2n_inv = 0
        for i in range(nD):
            thetan += pis[i] * Means[i]
            sigma2n_inv += pis[i] / Vars[i]
        sigma2n = 1/M/sigma2n_inv
        thetah = pm.Normal("thetah", mu=thetan, sigma=np.sqrt(sigma2n))

        Yobs = pm.Normal("Yobs", mu=thetah, sigma=np.sqrt(sigma2), observed=D)
    if diag:
        return {"UIPJS":UIPJS, "pis": pis}
    else:
        return UIPJS

def getLCPNormal(D, Ds):
    nD = len(Ds)
    Means = [np.mean(Dh) for Dh in Ds]
    Vars  = [np.var(Dh) for Dh in Ds]
    myModel = pm.Model()
    with myModel:
        logtaus = pm.Uniform("logtaus", lower=-30, upper=30, shape=nD)
        sigma2 = pm.InverseGamma("sigma2", alpha=0.01, beta=0.01)
        ws = [1/(Vars[jj]/len(Ds[jj]) + 1/np.exp(logtaus[jj])) for jj in range(nD)]
        ws = np.array(ws)
        thetan = np.sum(ws*Means)/np.sum(ws)
        sigma2n = 1/np.sum(ws)
        thetah = pm.Normal("thetah", mu=thetan, sigma=np.sqrt(sigma2n))

        Yobs = pm.Normal("Yobs", mu=thetah, sigma=np.sqrt(sigma2), observed=D)
    return myModel  

def getNPPNormal(D, Ds):
    nD = len(Ds)
    nDs = [len(Dh) for Dh in Ds]
    Means = [np.mean(Dh) for Dh in Ds]
    Vars  = [np.var(Dh) for Dh in Ds]
    myModel = pm.Model()
    with myModel:
        gammas = pm.Uniform("gammas", lower=0, upper=1, shape=nD)
        sigma2 = pm.InverseGamma("sigma2", alpha=0.01, beta=0.01)

        num_thetan = 0
        den_thetan = 0
        sigma2n_inv = 0
        for i in range(nD):
            num_thetan += gammas[i] * nDs[i] / Vars[i] * Means[i]
            den_thetan += gammas[i] * nDs[i] / Vars[i] 
            sigma2n_inv += gammas[i] * nDs[i] / Vars[i]
        thetan = num_thetan / den_thetan
        sigma2n = 1 / sigma2n_inv
        thetah = pm.Normal("thetah", mu=thetan, sigma=np.sqrt(sigma2n))

        Yobs = pm.Normal("Yobs", mu=thetah, sigma=np.sqrt(sigma2), observed=D)
    return myModel  
