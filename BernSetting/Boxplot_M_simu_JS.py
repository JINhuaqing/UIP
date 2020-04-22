import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from utilities_bern import *
import pickle
import argparse
from tqdm import tqdm
np.random.seed(1)


parser = argparse.ArgumentParser(description="UIP-JS boxplot")
parser.add_argument("-p0", type=float, default=0.4, help="Parameter of the current data")
args = parser.parse_args()

#p0s = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
p0 = args.p0
n = 60
ps = [0.25, 0.40]
ns = [40, 40]

Allres = []
numRep = 100
for idx in tqdm(range(1, numRep+1)):
    D0 = bernoulli.rvs(p0, size=n)
    Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]

    post_sps_UIPJS = gen_post_UIP_KL_MCMC(20000, D0, Ds, thin=10, burnin=10000)
    JSMsps = post_sps_UIPJS["sps_M"]
    Allres.append(JSMsps)

with open(f"./Boxplot_M_JS_Simu{numRep}_{int(p0*100)}.pkl", "wb") as f:
    pickle.dump(Allres, f)
