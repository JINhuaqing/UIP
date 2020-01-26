import numpy as np
from scipy.stats import norm
import pickle
import argparse
from utilities_norm import gen_post_full, gen_post_jef, getNPPcon, getLCPcon, getUIPJScon, getUIPDcon
import pymc3 as pm

np.random.seed(2020)

parser = argparse.ArgumentParser(description='UIP Normal')
parser.add_argument('-t0', type=float, default=0, help='mean of current data')
args = parser.parse_args()


Num = 1000
results = []
theta0 = args.t0
sigma0 = sigma1 = sigma2 = 1
n = 40
thetas = [0.5, 1]
ns = [50, 100]
print(f"The theta0 is {theta0}.")


for jj in range(Num):
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

    print("==" * 100)
    # NPP prior
    NPP_model = getNPPcon(D, Ds)
    with NPP_model:
        post_normal_NPP = pm.sample(draws=2500, tune=15000, target_accept=0.8, cores=4, chains=4)
    result["NPP"] = post_normal_NPP
    print("The results of NPP")
    diverging = post_normal_NPP['diverging']
    print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
    diverging_pct = diverging.nonzero()[0].size / len(post_normal_NPP) * 100
    print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))
    print(pm.summary(post_normal_NPP))

    print("--" * 100)
    # LCP prior
    LCP_model = getLCPcon(D, Ds)
    with LCP_model:
        post_normal_LCP = pm.sample(draws=2500, tune=15000, target_accept=0.9, cores=4, chains=4)
    result["LCP"] = post_normal_LCP
    print("The results of LCP")
    diverging = post_normal_LCP['diverging']
    print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
    diverging_pct = diverging.nonzero()[0].size / len(post_normal_LCP) * 100
    print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))
    print(pm.summary(post_normal_LCP))

    print("--" * 100)
    # UIPD prior
    UIPD_model = getUIPDcon(D, Ds)
    with UIPD_model:
        post_normal_UIPD = pm.sample(draws=2500,tune=15000, target_accept=0.8, cores=4, chains=4)
    result["UIPD"] = post_normal_UIPD
    print("The results of UIPD")
    diverging = post_normal_UIPD['diverging']
    print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
    diverging_pct = diverging.nonzero()[0].size / len(post_normal_UIPD) * 100
    print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))
    print(pm.summary(post_normal_UIPD))

    print("--" * 100)
    # UIPJS prior
    UIPJS_model = getUIPJScon(D, Ds)
    with UIPJS_model:
        post_normal_UIPJS = pm.sample(draws=2500, tune=15000, target_accept=0.8, cores=4, chains=4)
    result["UIPJS"] = post_normal_UIPJS
    print("The results of UIPJS")
    diverging = post_normal_UIPJS['diverging']
    print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
    diverging_pct = diverging.nonzero()[0].size / len(post_normal_UIPJS) * 100
    print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))
    print(pm.summary(post_normal_UIPJS))



    print(f"The iteration {jj+1}/{Num}."
        )
    results.append(result)


with open(f"MCMCNorm_{Num}_p0{int(100*theta0)}_n{int(n)}.pkl", "wb") as f:
    pickle.dump(results, f)
