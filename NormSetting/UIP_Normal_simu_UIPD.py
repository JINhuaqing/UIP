import numpy as np
from scipy.stats import norm
import pickle
import argparse
from utils_norm import *
import pymc3 as pm
import os.path as osp
import time
from pathlib import Path
from easydict import EasyDict as edict

np.random.seed(2020)

parser = argparse.ArgumentParser(description='UIP Normal')
parser.add_argument('-t0', type=float, default=0, help='mean of current data')
args = parser.parse_args()


Num = 1000
# theta0 from 0 to 0.5 step by 0.05
theta0 = args.t0
sigma0 = sigma1 = sigma2 = 1
n = 120
thetas = [0.5, 1]
ns = [100, 50]
print(f"The theta0 is {theta0}.")

dirname  = Path(f"./results/UIPDSIMR1_{int(100*theta0)}_n{int(n)}")
if not dirname.exists():
    dirname.mkdir()
results = list(dirname.glob("*.pkl"))
numRes = len(results)


init = time.time()
for jj in range(Num):
    result = edict()
    D = norm.rvs(loc=theta0, scale=sigma0, size=n)
    Ds = [ norm.rvs(loc=thetas[i], scale=sigma1, size=ns[i]) for i in range(len(ns))] 
    result["data"] = {"D":D, 
                      "Ds": Ds, 
                      "theta0": theta0,
                      "thetas": thetas,
                      "n" : n ,
                      "ns": ns
                      } 

    if jj >= numRes:
        print("==" * 100)
        ctm = np.round((time.time()-init)/60, 4)
        if jj-numRes >= 1:
            pctm = np.round(ctm/(jj-numRes), 4)
        else:
            pctm = ctm
        outstring = f"%%The iteration {jj+1}/{Num}. Totol time is {ctm} min, per iter time is {pctm} min.%%"
        print("%" * len(outstring))
        print(outstring)
        print("%" * len(outstring))


        print("--" * 100)
        # UIPD prior
        UIPD_model = getUIPDNormal(D, Ds)
        with UIPD_model:
            step = pm.Metropolis()
            post_normal_UIPD = pm.sample(draws=5000,tune=5000, 
                    step=step,
                    #target_accept=0.8, 
                    cores=4, chains=4)
        result["UIPD"] = post_normal_UIPD
        print("The results of UIPD")
        print(pm.summary(post_normal_UIPD))


        print(f"Saving results at iteration {jj+1}")
        with open(dirname/f"{jj+1}.pkl", "wb") as f:
            pickle.dump(result, f)
