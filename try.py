import numpy as np
import pymc3 as pm
from scipy.stats import norm


#np.random.seed(0)


n = 40
theta0 = 0
sigma0 = sigma1 = sigma2 = 1
thetas = [0.5, 1]
ns = [50, 100]
D = norm.rvs(loc=theta0, scale=sigma0, size=n)
Ds = [norm.rvs(loc=thetas[i], scale=sigma1, size=ns[i]) for i in range(len(ns))] 
D1, D2 = Ds

varh1, varh2 = np.var(D1), np.var(D2)
n1, n2 = len(D1), len(D2)
mh1, mh2 = np.mean(D1), np.mean(D2)

def getUIPDcon(D, Ds):
    nD = len(Ds)
    nsSum = np.sum([len(Dh) for Dh in Ds])
    Means = [np.mean(Dh) for Dh in Ds]
    Vars  = [np.var(Dh) for Dh in Ds]
    UIPDm = pm.Model()
    with UIPDm:
        pis = pm.Dirichlet("pis", np.ones(nD))
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
    return UIPDm

def getLCPcon(D, Ds):
    nD = len(Ds)
    nsSum = np.sum([len(Dh) for Dh in Ds])
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
        thetah = pm.Normal("thetat", mu=thetan, sigma=np.sqrt(sigma2n))
        #thetat = pm.Normal("thetat", mu=0, sigma=1)
        #thetah = pm.Deterministic("thetah", thetan + thetat * np.sqrt(sigma2n))
        Yobs = pm.Normal("Yobs", mu=thetah, sigma=np.sqrt(sigma2), observed=D)
    return myModel  


#UIPD = pm.Model()
#with UIPD:
#    pis = pm.Dirichlet("pis", np.ones(2))
#    sigma2 = pm.InverseGamma("sigma2", alpha=0.01, beta=0.01)
#    M = pm.Uniform("M", lower=0, upper=n1+n2)
#    
#    thetan = pis[0] * mh1 + pis[1] * mh2
#    sigma2n = 1/M/(pis[0]/varh1+pis[1]/varh2) 
#
#    thetah = pm.Normal("thetah", mu=thetan, sigma=np.sqrt(sigma2n))
#
#    Yobs = pm.Normal("Yobs", mu=thetah, sigma=np.sqrt(sigma2), observed=D)
#
#
def getUIPJScon(D, Ds):
    def KLnorm(mu1, mu2, sigma1, sigma2):
        itm1 = np.log(sigma2/sigma1)
        itm2 = (sigma1**2 + (mu2-mu1)**2)/(2*sigma2**2) - 0.5
        return itm1 + itm2
    def JSnorm(mu1, mu2, sigma1, sigma2):
        KL1 = KLnorm(mu1, mu2, sigma1, sigma2)
        KL2 = KLnorm(mu2, mu1, sigma2, sigma1)
        return (KL1 + KL2)/2

    nD = len(Ds)
    nsSum = np.sum([len(Dh) for Dh in Ds])
    Means = [np.mean(Dh) for Dh in Ds]
    Vars  = [np.var(Dh) for Dh in Ds]
    muinits = []
    varinits = []
    for idx, Dh in enumerate(Ds):
        varinits.append(1/(1/100 + len(Dh)/Vars[idx]))
        muinits.append(np.sum(Dh)/(len(Dh)+Vars[idx]/100))
    varinits.append(1/(1/100 + len(D)/np.var(D)))
    muinits.append(np.sum(D)/(len(D)+np.var(D)/100))
   
    invPis = [JSnorm(muinits[i], muinits[-1], varinits[i]**0.5,  varinits[-1]**0.5) for i in range(len(Ds))]
    Pis = 1/(np.array(invPis) + 1e-10)
    pis = np.array(Pis)/np.sum(Pis)

    UIPDJS = pm.Model()
    with UIPDJS:
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
    return UIPDJS

def getNPPcon(D, Ds):
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
            sigma2n_inv = gammas[i] * nDs[i] / Vars[i]
        thetan = num_thetan / den_thetan
        sigma2n = 1 / sigma2n_inv
        thetah = pm.Normal("thetah", mu=thetan, sigma=np.sqrt(sigma2n))

        Yobs = pm.Normal("Yobs", mu=thetah, sigma=np.sqrt(sigma2), observed=D)
    return myModel  
myModel = getLCPcon(D, Ds)
with myModel:
    trace = pm.sample(2500, tune=1000, cores=4, chains=4, target_accept=0.9)
print(trace["diverging"])
print(pm.summary(trace))
