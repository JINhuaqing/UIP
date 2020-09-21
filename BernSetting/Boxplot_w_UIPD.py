import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from utilities_bern import *
import pickle
from tqdm import tqdm
np.random.seed(1)


p0s = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
n = 40
ps = [0.2, 0.8]
ns = [40, 40]
numRep = 100

Allres = []
for idx in tqdm(range(1, numRep+1)):
    D0s = [bernoulli.rvs(p0s[i], size=n) for i in range(len(p0s))]
    Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]

    Dwss = {}
    DMss = {}
    for p0, D0 in zip(p0s, D0s):
        UIPD_model = getUIPDBern(D0, Ds, upM=n)
        with UIPD_model:
            step = pm.Metropolis()
            post_Bern_UIPD = pm.sample(draws=5000, tune=5000,
                    #target_accept=0.9, 
                    step=step,
                    cores=4, chains=4)
        Dws = post_Bern_UIPD["pis"]
        DMs = post_Bern_UIPD["M"]
        Dwss[f"{p0}"] = Dws
        DMss[f"{p0}"] = DMs
    Allres.append([Dwss, DMss])

with open(f"./Boxplot_w_UIPD{numRep}.pkl", "wb") as f:
    pickle.dump(Allres, f)
    
    
