from utilities_bern import *
import numpy as np
from scipy.stats import bernoulli
import scipy.stats as ss
import pymc3.stats as pymcs
import pickle
import argparse
np.random.seed(0)

parser = argparse.ArgumentParser(description='UIP Bernoulli')
parser.add_argument('-p0', type=float, default=0.4, help='successful probability of current data')

args = parser.parse_args()


def popu_beta(alp, bt, cutoff):
    eqInt = ss.beta.ppf([cutoff, 1-cutoff], a=alp, b=bt)
    post_mean = alp/(alp+bt)
    sps = ss.beta.rvs(a=alp, b=bt, size=10000)
    HPDInt = pymcs.hpd(sps)
    return {"eq":eqInt, "post_mean":post_mean, "HPD": HPDInt}

def samp_beta(sps, cutoff):
    eqInt = np.quantile(sps, q=[cutoff, 1-cutoff])
    post_mean = np.mean(sps)
    HPDInt = pymcs.hpd(sps)
    return {"eq":eqInt, "post_mean":post_mean, "HPD": HPDInt}
     



Num = 1000
results = []
p0 = args.p0
n = 50
ps = [0.3, 0.8]
ns = [40, 40]
cutoff = 0.025
print(f"The p0 is {p0}.")

for jj in range(Num):
    result = {}
    D = bernoulli.rvs(p0, size=n)
    Ds = [ bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))] 
    result["data"] = {"D":D, 
                      "Ds": Ds, 
                      "p0": p0,
                      "ps": ps,
                      "n" : n ,
                      "ns": ns
                      } 

   
    # Jeffrey prior  
    alp_jef = 0.5 + D.sum()
    bt_jef = 0.5 + len(D) - D.sum()
    res_jef = popu_beta(alp_jef, bt_jef, cutoff=cutoff)
    result["jef"] = res_jef

    # full borrowing 
    alp_full = 0.5 + D.sum() + np.sum([Dh.sum() for Dh in Ds])
    bt_full = 0.5 + len(D) - D.sum() + np.sum([len(Dh) - Dh.sum() for Dh in Ds])
    res_full = popu_beta(alp_full, bt_full, cutoff=cutoff)
    result["full"] = res_full

    # JPP  
    post_sps_jpp = gen_post_jpp(10000, D, Ds)
    res_jpp = samp_beta(post_sps_jpp["sps"], cutoff=cutoff)
    result["jpp"] = res_jpp

    #UIP-KL
    try:
        post_sps_UIPKL = gen_post_UIP_KL(10000, D, Ds)
        res_UIPKL = samp_beta(post_sps_UIPKL["sps"], cutoff=cutoff)
    except Exception as e:
        post_sps_UIPKL = {"sps": []}
        res_UIPKL = {}
        print(e)
    result["UIPKL"] = res_UIPKL
    result["UIPKL_sps"] = post_sps_UIPKL

    #UIP-multi
    try:
        post_sps_UIPm = gen_post_UIP_multi(10000, D, Ds)
        res_UIPm = samp_beta(post_sps_UIPm["sps"], cutoff=cutoff)
    except Exception as e:
        post_sps_UIPm = {"sps": []}
        res_UIPm = {}
        print(e)
    result["UIPm"] = res_UIPm
    result["UIPm_sps"] = post_sps_UIPm


    print(f"The iteration {jj+1}/{Num}."
          f"Number of samples for UIPKL is {len(post_sps_UIPKL['sps'])}." 
          f"Number of samples for UIPm is {len(post_sps_UIPm['sps'])}." 
        )
    results.append(result)

with open(f"Bern_Num{Num}_p0{int(100*p0)}_n{int(n)}.pkl", "wb") as f:
    pickle.dump(results, f)
