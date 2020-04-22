import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from utilities_bern import *
import pickle
from tqdm import tqdm
np.random.seed(1)



p0s = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
n = 60
ps = [0.25, 0.40]
ns = [40, 40]

Allres = []
numRep = 100
for idx in tqdm(range(1, numRep+1)):
    D0s = [bernoulli.rvs(p0s[i], size=n) for i in range(len(p0s))]
    Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]

    DMspss = {}
    for p0, D0 in zip(p0s, D0s):
        post_sps_UIPD = gen_post_UIP_D_MCMC(60000, D0, Ds, thin=50, burnin=10000)
        DMsps = post_sps_UIPD["sps_M"]
        DMspss[f"{p0}"] = DMsps
    Allres.append(DMspss)

with open(f"./Boxplot_M_D_Simu{numRep}.pkl", "wb") as f:
    pickle.dump(Allres, f)
