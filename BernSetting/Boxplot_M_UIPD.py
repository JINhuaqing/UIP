import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from utils_bern import *
import pickle
from tqdm import tqdm
np.random.seed(1)



p0s = np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
n = 40
ps = [0.10, 0.30]
ns = [40, 40]

Allres = []
numRep = 100
for idx in tqdm(range(1, numRep+1)):
    D0s = [bernoulli.rvs(p0s[i], size=n) for i in range(len(p0s))]
    Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]

    DMspss = {}
    for p0, D0 in zip(p0s, D0s):
        UIPD_model = getUIPDBern(D0, Ds, upM=n)
        with UIPD_model:
            step = pm.Metropolis()
            post_Bern_UIPD = pm.sample(draws=5000, tune=5000,
                    #target_accept=0.9, 
                    step=step,
                    cores=4, chains=4)
        DMsps = post_Bern_UIPD["M"]
        DMspss[f"{p0}"] = DMsps
    Allres.append(DMspss)

with open(f"./Boxplot_M_UIPD{numRep}.pkl", "wb") as f:
    pickle.dump(Allres, f)
