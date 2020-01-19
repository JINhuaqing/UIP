import numpy as np
import numpy.random as npr
from scipy.stats import norm, invgamma 


def sigma2cond_theta(theta, D):
    n = len(D)
    shape = n/2
    scale = np.sum((D-theta)**2)/2
    return invgamma.rvs(a=shape, scale=scale, size=1)
    

def thetacond_sigma2(sigma2, D):
    n = len(D)
    loc = np.mean(D)
    scale = np.sqrt(sigma2/n) 
    return norm.rvs(loc=loc, scale=scale, size=1)


def gen_post_jef(N, D, burnin=5000, thin=1, diag=False):
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



def gen_post_full(N, D, Ds, burnin=5000, thin=1, diag=False):
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
