import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from utilities_bern import *
import pickle
from tqdm import tqdm
np.random.seed(1)


p0s = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
n = 60
ps = [0.3, 0.8]
ns = [40, 40]
numRep = 100

Allres = []
for idx in tqdm(range(1, numRep+1)):
    D0s = [bernoulli.rvs(p0s[i], size=n) for i in range(len(p0s))]
    Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]



    dspss = {}
    for p0, D0 in zip(p0s, D0s):
        post_sps_UIPD = gen_post_UIP_D_MCMC(60000, D0, Ds, thin=50, burnin=10000)
        mulsps = post_sps_UIPD["sps_m"]
        dsps = mulsps/(mulsps.sum(axis=1).reshape(-1, 1))
        dspss[f"{p0}"]=dsps
    Allres.append(dspss)

with open(f"./Boxplot_pi_Simu{numRep}.pkl", "wb") as f:
    pickle.dump(Allres, f)
    
    
