import numpy as np
import numpy.random as npr
from scipy.stats import norm, invgamma 


def sigma2cond_theta(theta, **kwargs):
    D = kwargs['D']
    if "Ds" in kwargs.keys():
        Dhall = np.concatenate(Ds)
        D = np.concatenate((D, Dhall))
    n = len(D)
    shape = n/2
    scale = np.sum((D-theta)**2)/2
    return invgamma.rvs(a=shape, scale=scale, size=1)
    

def thetacond_sigma2(sigma2, **kwargs):
    D = kwargs['D']
    if "Ds" in kwargs.keys():
        Dhall = np.concatenate(Ds)
        D = np.concatenate((D, Dhall))
    n = len(D)
    loc = np.mean(D)
    scale = np.sqrt(sigma2/n) 
    return norm.rvs(loc=loc, scale=scale, size=1)

