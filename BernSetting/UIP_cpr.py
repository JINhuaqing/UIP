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
n = 120
ps = [0.25, 0.4]
ns = [50, 100]
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
    result["jef_popu"] = [alp_jef, bt_jef]

    # full borrowing 
    alp_full = 0.5 + D.sum() + np.sum([Dh.sum() for Dh in Ds])
    bt_full = 0.5 + len(D) - D.sum() + np.sum([len(Dh) - Dh.sum() for Dh in Ds])
    res_full = popu_beta(alp_full, bt_full, cutoff=cutoff)
    result["full"] = res_full
    result["full_popu"] = [alp_full, bt_full]

    # JPP  
    post_sps_jpp = gen_post_jpp(50000, D, Ds)
    res_jpp = samp_beta(post_sps_jpp["sps"], cutoff=cutoff)
    result["jpp"] = res_jpp
    result["jpp_sps"] = post_sps_jpp

    #UIP-KL
    try:
        post_sps_UIPKL = gen_post_UIP_KL_MCMC(20000, D, Ds, burnin=4000, thin=8)
        #post_sps_UIPKL = gen_post_UIP_KL(20000, D, Ds, Maxiter=100)
        res_UIPKL = samp_beta(post_sps_UIPKL["sps"], cutoff=cutoff)
    except Exception as e:
        post_sps_UIPKL = {"sps": []}
        res_UIPKL = {}
        print(e)
    result["UIPKL"] = res_UIPKL
    result["UIPKL_sps"] = post_sps_UIPKL

    #UIP-multi
    try:
        post_sps_UIPD = gen_post_UIP_D_MCMC(20000, D, Ds, burnin=4000, thin=8)
        #post_sps_UIPD = gen_post_UIP_D(20000, D, Ds, Maxiter=100)
        res_UIPD = samp_beta(post_sps_UIPD["sps"], cutoff=cutoff)
    except Exception as e:
        post_sps_UIPD = {"sps": []}
        res_UIPD = {}
        print(e)
    result["UIPD"] = res_UIPD
    result["UIPD_sps"] = post_sps_UIPD


    print(f"The iteration {jj+1}/{Num}."
          f"Number of samples for UIPKL is {len(post_sps_UIPKL['sps'])}." 
          f"Number of samples for UIPm is {len(post_sps_UIPD['sps'])}." 
        )
    results.append(result)

#with open(f"RJBern_Num{Num}_p0{int(100*p0)}_n{int(n)}.pkl", "wb") as f:
with open(f"MCMCBern_nsdiff_Num{Num}_p0{int(100*p0)}_n{int(n)}.pkl", "wb") as f:
    pickle.dump(results, f)
