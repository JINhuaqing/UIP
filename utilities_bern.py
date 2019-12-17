import numpy as np
import numpy.random as npr
from scipy.special import beta as Beta
from scipy.stats import poisson, uniform, multinomial


# compute corresponding (alpha, beta) given M1, M2,...
def cond_prior(Ds, Ms):
    """
    Ds: historical datasets, list of array or list 
    Ms: vector of Mi, array

    """
    M = Ms.sum()
    MLEs = np.array([np.mean(Dh) for Dh in Ds])
    av = np.sum(Ms*MLEs)/M
    Is = 1/MLEs/(1-MLEs)
    bv = 1/np.sum(Ms*Is)
    alpha = av * (av*(1-av)/bv - 1)
    beta = (1-av) * (av*(1-av)/bv - 1)
    return [alpha, beta]


## UIP-KL prior functions under Bernoulli 

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

def gen_trunc_pois(N, lam, low, up):
    # generate Poi(lam) between (low, up]
    cutoff1, cutoff2 = poisson.cdf(low, lam), poisson.cdf(up, lam)
    usps = uniform.rvs(loc=cutoff1, scale=cutoff2-cutoff1, size=N)
    return poisson.ppf(usps, lam)
    


# generate prior from UIP-KL prior
def gen_prior_UIP_KL(N, D, Ds, lam):
    paras = [[1+np.sum(Dh), 1+len(Dh)-np.sum(Dh)] for Dh in Ds]
    para0 = [1+np.sum(D), 1+len(D)-np.sum(D)]
    ms = np.array([JS_dist_beta(para0, para) for para in paras])
    ms = 1 / (ms + 1e-10) # add 1e-10 to avoid 0 
    sps_poi = gen_trunc_pois(N, lam, 0, 2*lam)
    Mss = [ms*sp_poi/ms.sum() for sp_poi in sps_poi]
    condpriors = [cond_prior(Ds, Ms) for Ms in Mss]
    sps = [npr.beta(condprior[0], condprior[1], 1)[0] for condprior in condpriors]
    sps = np.array(sps)
    return {"sps": sps, "sps_poi": sps_poi}


# genrate sample from posteior given D with UPDKL
def gen_post_UIP_KL(N, D, Ds, fct=0.5, Maxiter=50, Ns=10000):
    MLE, n, Dsum, nDsum = np.mean(D), len(D), np.sum(D), len(D) - np.sum(D)
    logden = Dsum * np.log(MLE) + nDsum * np.log(1-MLE)
    den = np.exp(logden)
    sps_full = []
    sps_poi_full = []
    flag = 1

    while len(sps_full) <= N:
        allsps = gen_prior_UIP_KL(Ns, D, Ds, n*fct)
        sps = allsps["sps"]
        lognums = Dsum * np.log(sps) + nDsum * np.log(1-sps)
        nums = np.exp(lognums)

        usps = uniform.rvs(size=Ns)
        vs = nums/den
        keepidx = vs >= usps
        sps_full = sps_full + list(sps[keepidx])
        sps_poi_full = sps_poi_full + list(allsps["sps_poi"][keepidx])
        flag += 1
        if flag > Maxiter:
            break
    return {"sps": np.array(sps_full), "sps_poi": np.array(sps_poi_full)}


## UIP-M prior functions
# generate sample from prior 
def gen_prior_UIP_multi(N, Ds, lam):
    numDs = len(Ds)
    sps_poi = gen_trunc_pois(N, lam, 0, 2*lam)
    sps_mul = [multinomial.rvs(sp_poi, np.ones(numDs)/numDs, 1)[0] for sp_poi in sps_poi] 
    condpriors = [cond_prior(Ds, sp_mul) for sp_mul in sps_mul] 
    sps = [npr.beta(condprior[0], condprior[1], 1)[0] for condprior in condpriors]
    sps, sps_mul = np.array(sps), np.array(sps_mul)
    return {"sps": sps, "sps_poi": sps_poi, "sps_mul": sps_mul}

# genrate sample from posteior given D
def gen_post_UIP_multi(N, D, Ds, fct=0.5, Maxiter=50, Ns=10000):
    MLE, n, Dsum, nDsum = np.mean(D), len(D), np.sum(D), len(D) - np.sum(D)
    logden = Dsum * np.log(MLE) + nDsum * np.log(1-MLE)
    den = np.exp(logden)
    sps_full = []
    sps_poi_full = []
    sps_mul_full = []
    flag = 1

    while len(sps_full) <= N:
        allsps = gen_prior_UIP_multi(Ns, Ds, n*fct)
        sps = allsps["sps"]
        lognums = Dsum * np.log(sps) + nDsum * np.log(1-sps)
        nums = np.exp(lognums)

        usps = uniform.rvs(size=Ns)
        vs = nums/den
        keepidx = vs >= usps
        sps_full = sps_full + list(sps[keepidx])
        sps_poi_full = sps_poi_full + list(allsps["sps_poi"][keepidx])
        sps_mul_full = sps_mul_full + list(allsps["sps_mul"][keepidx])
        flag += 1
        if flag > Maxiter:
            break
    return {"sps": np.array(sps_full), "sps_poi": np.array(sps_poi_full), "sps_mul": np.array(sps_mul_full)}



## JPP priors funtions
def gen_conpostp_jpp(D, Ds, gammas):
    nDs = len(Ds)
    alp = np.sum(D) + np.sum([gammas[i]*np.sum(Ds[i]) for i in range(nDs)])
    bt = len(D) - np.sum(D) + np.sum([gammas[i]*(len(Ds[i]) - np.sum(Ds[i])) for i in range(nDs)])
    return npr.beta(alp, bt, 1)

def gen_conpostga_jpp(n, p, Dh):
    a, b = np.sum(Dh), len(Dh) - np.sum(Dh)
    sps_uni = uniform.rvs(size=n)
    den = b * np.log(1-p) + a * np.log(p)
    nums = np.log(1 + ((1-p)**b * p**a - 1) * sps_uni)
    return nums/den

def gen_post_jpp(N, D, Ds, burnin=5000):
    flag = 0
    sps = []
    gammass = []
    p = 0.1
    for i in range(N+burnin):
        gammas = [gen_conpostga_jpp(1, p, Dh) for  Dh in Ds] 
        p = gen_conpostp_jpp(D, Ds, gammas)
        if i > burnin:
            sps.append(p)
            gammass.append(gammas)
    sps, gammass = np.array(sps), np.array(gammass)
    return {"sps":sps, "gam":gammass}


