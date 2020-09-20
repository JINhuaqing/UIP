from utils_bern import *
import numpy as np
from scipy.stats import bernoulli
import scipy.stats as ss
import pymc3.stats as pymcs
import pymc3 as pm
import pickle
from pathlib import Path
import argparse
import time
from easydict import EasyDict as edict
np.random.seed(0)

parser = argparse.ArgumentParser(description='UIP Bernoulli')
parser.add_argument('-p0', type=float, default=0.2, help='successful probability of current data')

args = parser.parse_args()

Num = 1000
# from 0.2 to 0.5, step by 0.05 
p0 = args.p0
n = 120
ps = [0.2, 0.4]
ns = [50, 100]
print(f"The p0 is {p0}.")

dirname  = Path(f"./results/BiostatR1_{int(100*p0)}_n{int(n)}")
if not dirname.exists():
    dirname.mkdir()
results = list(dirname.glob("*.pkl"))
numRes = len(results)
init = time.time()

for jj in range(Num):
    print("="*50)
    print(f"The iteration {jj+1}/{Num}.")
    result = edict()
    D = bernoulli.rvs(p0, size=n)
    Ds = [ bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))] 
    result["data"] = {"D":D, 
                      "Ds": Ds, 
                      "p0": p0,
                      "ps": ps,
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

        # Jeffrey prior  
        alp_jef = 0.5 + D.sum()
        bt_jef = 0.5 + len(D) - D.sum()
        result["jef"] = [alp_jef, bt_jef]

        # full borrowing 
        alp_full = 0.5 + D.sum() + np.sum([Dh.sum() for Dh in Ds])
        bt_full = 0.5 + len(D) - D.sum() + np.sum([len(Dh) - Dh.sum() for Dh in Ds])
        result["full"] = [alp_full, bt_full]

        print("*"*50)
        # UIP-JS
        UIPJS_model = getUIPJSBern(D, Ds)
        with UIPJS_model:
            step = pm.Metropolis()
            post_Bern_UIPJS = pm.sample(draws=5000, tune=5000, 
                    #target_accept=0.9, 
                    step=step,
                    cores=4, chains=4)
        result["UIPJS"] = post_Bern_UIPJS

        print("*"*50)
        # UIP-D
        UIPD_model = getUIPDBern(D, Ds)
        with UIPD_model:
            step = pm.Metropolis()
            post_Bern_UIPD = pm.sample(draws=5000, tune=5000,
                    #target_accept=0.9, 
                    step=step,
                    cores=4, chains=4)
        result["UIPD"] = post_Bern_UIPD

        print("*"*50)
        # NPP
        NPP_model = getNPPBern(D, Ds)
        with NPP_model:
            step = pm.Metropolis()
            post_Bern_NPP = pm.sample(draws=5000, tune=5000,
                    #target_accept=0.8, 
                    step=step,
                    cores=4, chains=4)
        result["NPP"] = post_Bern_NPP

        print("*"*50)
        # rMAP
        rMAPprior = getMixBeta(Ds)
        rMAPpost = MixPostBeta(rMAPprior, D)
        post_Bern_rMAP = genMixBeta(20000, rMAPpost)
        result["rMAP"] = post_Bern_rMAP


        print(f"Saving results at iteration {jj+1}")
        with open(dirname/f"{jj+1}.pkl", "wb") as f:
            pickle.dump(result, f)


