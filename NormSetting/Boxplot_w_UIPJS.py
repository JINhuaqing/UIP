import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from utils_norm import *
import pickle
from tqdm import tqdm
np.random.seed(1)


theta0s = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
n = 40
thetas = [-0.30, 0.30]
ns = [40, 40]
sigma0 = sigma1 = sigma2 = 1
numRep = 100

Allres = []
for idx in tqdm(range(1, numRep+1)):
    D0s = [norm.rvs(loc=theta0s[i], scale=sigma0, size=n) for i in range(len(theta0s))]
    Ds = [norm.rvs(loc=thetas[i], scale=sigma1, size=ns[i]) for i in range(len(ns))]

    Dwss = {}
    DMss = {}
    for theta0, D0 in zip(theta0s, D0s):
        res = getUIPJSNormal(D0, Ds, upM=n, diag=True)
        UIPJS_model = res["UIPJS"] 
        with UIPJS_model:
            step = pm.Metropolis()
            post_norm_UIPJS = pm.sample(draws=5000, tune=5000,
                    #target_accept=0.9, 
                    step=step,
                    cores=4, chains=4)
        Dws = res["pis"]
        DMs = post_norm_UIPJS["M"]
        Dwss[f"{theta0}"] = Dws
        DMss[f"{theta0}"] = DMs
    Allres.append([Dwss, DMss])

with open(f"./Boxplot_w_UIPJS{numRep}.pkl", "wb") as f:
    pickle.dump(Allres, f)
    
    
