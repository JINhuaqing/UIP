import numpy as np
import numpy.random as npr
from scipy.stats import beta, norm, bernoulli
from scipy.special import roots_legendre, roots_hermite
from numpy.random import dirichlet
from scipy.special import beta as Beta
from tqdm import tqdm_notebook as tqdm
from easydict import EasyDict as edict
import multiprocessing as mp



# Function to draw sample for theta parameter for binary data with MCMC sampling
def MH_beta(n, target_density, burnin):
    """
    Input:
        n: The number of samples
        target_density: The target density fn to draw
        burnin: The number of burn-in samples
    Return:
        The samples, array, n x 1
    """
    
    # The transition density function
    def trans_den(xt, yt):
        if yt == 0:
            if xt == 0:
                return np.Inf
            else:
                return 0
        elif yt == 1:
            if xt == 1:
                return np.Inf
            else:
                return 0
        else:
            return beta.pdf(xt, 10*yt, 10*(1-yt))
            
    xt = np.array([0.5])
    sps = []
    for i in range(n+burnin):
        if xt == 1:
            xt_can = 1
        elif xt == 0:
            xt_can = 0
        else:
            xt_can = beta.rvs(10*xt, 10*(1-xt), size=1)
            
        acc_prob = target_density(xt_can)*trans_den(xt, xt_can)/target_density(xt)/trans_den(xt_can, xt)
        acc_prob = np.min((acc_prob, 1))
        if npr.rand(1) < acc_prob:
            xt = xt_can
        if i >= burnin:
            sps.append(xt)
    return np.array(sps).reshape(-1)


# Function to draw sample for mean parameter for continuous data with MCMC sampling
def MH_Gaussian(n, target_density, burnin):
    """
    Input:
        n: The number of samples
        target_density: The target density fn to draw
        burnin: The number of burn-in samples
    Return:
        The samples, array, n x 1
    """
    # The transition density function
    def trans_den(xt, yt):
        return norm.pdf(xt, loc=yt, scale=1)
    xt = np.array([0])
    sps = []
    for i in range(n+burnin):
        xt_can = norm.rvs(loc=xt, scale=1, size=1)
        acc_prob = target_density(xt_can)*trans_den(xt, xt_can)/target_density(xt)/trans_den(xt_can, xt)
        acc_prob = np.min((acc_prob, 1))
        if npr.rand(1) < acc_prob:
            xt = xt_can
        if i >= burnin:
            sps.append(xt)
    return np.array(sps).reshape(-1)


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


# The conditional ESS 
def ESSBetaCon(M, ns, ps, n, p):
    """
    Input:
        M: The amount parameters
        ns: The number of sample size for each historical dataset
        ps: The mean parameters for each historical dataset
        p: The mean parameters for the current dataset
        n: The sample size of the current dataset
    Return:
        The ESS under the UIP-JS method
    
    """
    Ds = [ bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]
    D = bernoulli.rvs(p, size=n)
    msInv = []
    parasc = [1+np.sum(D), 1+len(D)-np.sum(D)]
    for i, nh in enumerate(ns):
        if nh <= n:
            parash = [1+np.sum(Ds[i]), 1+len(Ds[i])-np.sum(Ds[i])]
            msInv.append(JS_dist_beta(parasc, parash))
        else:
            JSs = []
            for j in range(100):
                Dhc = np.random.choice(Ds[i], size=n, replace=False)
                parashj = [1+np.sum(Dhc), 1+n-np.sum(Dhc)]
                JSs.append(JS_dist_beta(parasc, parashj))
            msInv.append(np.mean(JSs))
    ms = 1/(np.array(msInv)+1e-10)
    ws = ms/ms.sum()
            
            
    phats = np.array([np.mean(Dh) for Dh in Ds])
    mu = np.sum(ws * phats)
    invEta2 = M*np.sum(ws / (1-phats)/phats)
    ESS = mu*(1-mu)*invEta2-1
    return ESS


class ESSBetaH():
    def __init__(self, Ds, n, M, C=100):
        self.ns = np.array([len(Dh) for Dh in Ds])
        self.n = n
        self.M = M
        self.Ds = Ds
        self.C = C
        
        self.thetaBar = None
        self.prior12_sps = None
    
    def prior1_paras(self, wss):
        phats = np.array([np.mean(Dh) for Dh in self.Ds])
        mus = np.sum(wss * phats, axis=1)
        invEta2s = self.M*np.sum(wss / (1-phats)/phats, axis=1)
        alps = mus * (mus*(1-mus)*invEta2s-1)
        bets = (1-mus) * (mus*(1-mus)*invEta2s-1)
        return alps, bets
        
    def _prior1(self, theta, wss):
        alps, bets = self.prior1_paras(wss)
        return beta.pdf(theta, a=alps, b=bets)
        
    
    def _delta_prior1(self, theta, wss):
        alp_deltas, bet_deltas = self.prior1_paras(wss)
        alp_deltas, bet_deltas = alp_deltas/self.C, bet_deltas/self.C
        return beta.pdf(theta, a=alp_deltas, b=bet_deltas)
    
    def delta_prior12(self, theta):
        dalps = self.ns/self.n
        dalps[dalps>=1] = 1
        wss = dirichlet(dalps, size=10000)
        vs = self._delta_prior1(theta, wss)
        return np.mean(vs)
        
    def prior12(self, theta):
        dalps = self.ns/self.n
        dalps[dalps>=1] = 1
        wss = dirichlet(dalps, size=10000)
        vs = self._prior1(theta, wss)
        return np.mean(vs)
    
    def __delta_prior12_rvs(self, num):
        sps = MH_beta(n=num, target_density=self.delta_prior12, burnin=5000)
        return sps
    
    def __prior12_rvs(self, num):
        sps = MH_beta(n=num, target_density=self.prior12, burnin=5000)
        self.prior12_sps = sps
        return sps
    
    def deltaPost12(self, theta, ESS):
        if self.thetaBar is None:
            def func(theta):
                return theta*self.prior12(theta)
            self.thetaBar = self.integrate(func)
        logItm1 = np.log(self.delta_prior12(theta))
        logItm2 = ESS*self.thetaBar*np.log(theta)
        logItm3 = ESS*(1-self.thetaBar)*np.log(1-theta)
        logDen = logItm1 + logItm2 + logItm3
        return np.exp(logDen)
    
    def __delta_post12_rvs(self, num, ESS):
        def tdensity(theta):
            return self.deltaPost12(theta, ESS)
        sps = MH_beta(n=num, target_density=tdensity, burnin=5000)
        return sps
        
        
    def integrate(self, fun, a=0, b=1, nNodes=100):
        pts, ws = roots_legendre(nNodes)
        nPts = (b-a)/2*pts + (a+b)/2
        fvs = np.array([fun(nPt) for nPt in nPts])
        return np.sum(fvs*ws)*(b-a)/2
    
    
    def varPrior12(self):
        if self.thetaBar is None:
            def func(theta):
                return theta*self.prior12(theta)
            self.thetaBar = self.integrate(func)
        def funcSecMom(theta):
            return theta**2*self.prior12(theta)
        thetaSecMom = self.integrate(funcSecMom)
        return thetaSecMom-self.thetaBar**2
    
    def varDeltaPost12(self, ESS):
        def conFunc(theta):
            return self.deltaPost12(theta, ESS)
        Cons = self.integrate(conFunc)
        def deltaPost12Mean(theta):
            return self.deltaPost12(theta, ESS)*theta
        def deltaPost12SecMom(theta):
            return self.deltaPost12(theta, ESS)*theta**2
        secMom = self.integrate(deltaPost12SecMom)/Cons
        mean = self.integrate(deltaPost12Mean)/Cons
        return secMom - mean**2
        
    def getESS(self, MaxM=None):
        if MaxM is None:
            MaxM = int(1.5*self.M)
        varPrior = self.varPrior12()
        pool = mp.Pool(processes=10)
        esss = np.arange(1, MaxM+1)
        with pool as p:
            varPosts = p.map(self.varDeltaPost12, esss)
        #for ess in tqdm(range(1, MaxM+1), desc="ESS"):
        #    varPosts.append(self.varDeltaPost12(ess))
        varPosts = np.array(varPosts)
        diffs = np.abs(1/varPosts - 1/varPrior)
        idx = np.argmin(diffs)
        res = {"diffs":diffs, "ESS":idx+1}
        self.res = edict(res)
        
        

def ESSNormalCon(M, Ds, D):
    """
    Input:
        M: The amount parameters
        Ds: The historical datasets
        D: The current dataset
    Return:
        The ESS 
    
    """
    msInv = []
    n = len(D)
    parasc = obtainInitPost(D, n)
    for i, Dh in enumerate(Ds):
        parash = obtainInitPost(Dh, n)
        msInv.append(JSnorm(parasc[0], parash[0], parasc[1]**0.5, parash[1]**0.5))
    ms = 1/(np.array(msInv)+1e-10)
    ws = ms/ms.sum()
           
    varhats = np.array([np.var(Dh) for Dh in Ds])
    invEta2 = M*np.sum(ws / varhats) 
    ESS = np.var(D)*invEta2
    return ESS


class ESSNormalH(ESSBetaH):
    def __init__(self, Ds, D, M, C=100):
        self.ns = np.array([len(Dh) for Dh in Ds])
        self.n = len(D)
        self.M = M
        self.Ds = Ds
        self.D = D
        self.var = np.var(D)
        self.C = C
        
        self.thetaBar = None
        
    def prior1_paras(self, wss):
        thetas = np.array([np.mean(Dh) for Dh in self.Ds])
        varrs = np.array([np.var(Dh) for Dh in self.Ds])
        mus = np.sum(wss * thetas, axis=1)
        invEta2s = self.M*np.sum(wss / varrs, axis=1)
        return mus, invEta2s
        
    def _prior1(self, theta, wss):
        mus, invEta2s = self.prior1_paras(wss)
        etas = np.sqrt(1/invEta2s)
        return norm.pdf(theta, loc=mus, scale=etas)
        
    
    def _delta_prior1(self, theta, wss):
        mus, invEta2s = self.prior1_paras(wss)
        etas = np.sqrt(1/invEta2s)
        eta_deltas = etas * self.C
        return norm.pdf(theta, loc=mus, scale=eta_deltas)
    
    def integrate(self, fun, nNodes=100):
        def func(x):
            return fun(x)*np.exp(x**2)
        pts, ws = roots_hermite(nNodes)
        fvs = np.array([func(pt) for pt in pts])
        return np.sum(fvs*ws)
    
    def deltaPost12(self, theta, ESS):
        if self.thetaBar is None:
            def func(theta):
                return theta*self.prior12(theta)
            self.thetaBar = self.integrate(func)
        logItm1 = np.log(self.delta_prior12(theta))
        logItm2 = -ESS*(self.thetaBar-theta)**2/2/self.var
        logDen = logItm1 + logItm2
        return np.exp(logDen)
    
def KLnorm(mu1, mu2, sigma1, sigma2):
    itm1 = np.log(sigma2/sigma1)
    itm2 = (sigma1**2 + (mu2-mu1)**2)/(2*sigma2**2) - 0.5
    return itm1 + itm2

def JSnorm(mu1, mu2, sigma1, sigma2):
    KL1 = KLnorm(mu1, mu2, sigma1, sigma2)
    KL2 = KLnorm(mu2, mu1, sigma2, sigma1)
    return (KL1 + KL2)/2

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