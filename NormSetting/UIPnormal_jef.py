import numpy as np
from scipy.stats import norm
import pickle
import argparse
from utilities_norm import gen_post_full, gen_post_jef, getNPPcon, getLCPcon, getUIPJScon, getUIPDcon
import pymc3 as pm
import os.path as osp
import time
from pathlib import Path

np.random.seed(2020)

parser = argparse.ArgumentParser(description='UIP Normal')
parser.add_argument('-t0', type=float, default=0, help='mean of current data')
args = parser.parse_args()


Num = 1000
theta0 = args.t0
sigma0 = sigma1 = sigma2 = 1
n = 120
thetas = [0.5, 1]
ns = [50, 100]
print(f"The theta0 is {theta0}.")
results = []


init = time.time()
for jj in range(Num):
    print(f"{jj}/{Num}")
    result = {}
    D = norm.rvs(loc=theta0, scale=sigma0, size=n)
    Ds = [ norm.rvs(loc=thetas[i], scale=sigma1, size=ns[i]) for i in range(len(ns))] 
    result["data"] = {"D":D, 
                      "Ds": Ds, 
                      "theta0": theta0,
                      "thetas": thetas,
                      "n" : n ,
                      "ns": ns
                      } 

    # jeffrey's prior
    post_normal_jef = gen_post_jef(55000, D=D)
    result["jef"] = post_normal_jef
    # full borrowing under jeffrey's prior
    post_normal_full = gen_post_full(55000, D=D, Ds=Ds)
    result["full"] = post_normal_full

    results.append(result)




print(f"Saving results at iteration {jj+1}")
with open(f"Jeffrey_Full_{int(theta0*100)}.pkl", "wb") as f:
    pickle.dump(results, f)
