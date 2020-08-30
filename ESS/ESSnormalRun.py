from utils import ESSNormalCon, ESSNormalH
from scipy.stats import norm
from numpy.random import uniform
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import pickle
npr.seed(2020)

ns = [80, 100, 120]
n = 100
Ms = np.arange(10, 200, step=10)
thetas_range = [-0.5, 0.5]
sigmas_range = [0.9, 1.1]
sigmass = uniform(sigmas_range[0], sigmas_range[1], size=(50, 3))
thetass = uniform(thetas_range[0], thetas_range[1], size=(50, 3))
sigma0 = 1
theta0 = 0


ESSsM = []
for M in Ms:
    ESSs = []
    for i in tqdm(range(len(thetass)), desc=f"{M}/190"):
        thetas, sigmas = thetass[i], sigmass[i]
        D = norm.rvs(loc=theta0, scale=sigma0, size=n)
        Ds = [norm.rvs(loc=thetas[i], scale=sigmas[i], size=ns[i]) for i in range(len(ns))]
        ESS2 = ESSNormalH(Ds, D, M=M, C=1e6)
        ESS2.getESS()
        ESSs.append(ESS2.res)
    ESSsM.append(ESSs)
    

with open("./normal_ESS_UIPD.pkl", "wb") as f:
    pickle.dump(ESSsM, f)
