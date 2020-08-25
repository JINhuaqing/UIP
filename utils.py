import numpy as np
import numpy.random as npr
from scipy.stats import beta
from scipy.stats import norm



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