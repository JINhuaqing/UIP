import numpy as np
from scipy.stats import norm
import pickle
import argparse
import pystan
from utilities_norm import sigma2cond_theta, thetacond_sigma2, gen_post_full, gen_post_jef

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

sm_NPP = pystan.StanModel(file="./Normalstan/normNPP.stan", model_name="NPP_model")
sm_LCP = pystan.StanModel(file="./Normalstan/normLCP.stan", model_name="LCP_model")
sm_UIPD = pystan.StanModel(file="./Normalstan/normUIPD.stan", model_name="UIPD_model")
sm_UIPJS = pystan.StanModel(file="./Normalstan/normUIPJS.stan", model_name="UIPJS_model")

numc = 5

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
    data = {}
    data['n'] = n
    data['n1'] = ns[0]
    data['n2'] = ns[1]
    data['D'] = D
    data['D1'] = Ds[0]
    data['D2'] = Ds[1]


    # jeffrey's prior
    post_normal_jef = gen_post_jef(15000, D=D)
    result["jef"] = post_normal_jef

    # full borrowing under jeffrey's prior
    post_normal_full = gen_post_full(15000, D=D, Ds=Ds)
    result["full"] = post_normal_full

    # NPP prior
    fit_NPP = sm_NPP.sampling(data=data, chains=numc, iter=20000, warmup=10000, thin=numc, init="random")
    post_normal_NPP = fit_NPP.extract()
    result["NPP"] = post_normal_NPP

    # LCP prior
    fit_LCP = sm_LCP.sampling(data=data, chains=numc, iter=20000, warmup=10000, thin=numc, init="random")
    post_normal_LCP = fit_LCP.extract()
    result["LCP"] = post_normal_LCP

    # UIPD prior
    fit_UIPD = sm_UIPD.sampling(data=data, chains=numc, iter=20000, warmup=10000, thin=numc, init="random")
    post_normal_UIPD = fit_UIPD.extract()
    result["UIPD"] = post_normal_UIPD

    # UIPJS prior
    fit_UIPJS = sm_UIPJS.sampling(data=data, chains=numc, iter=20000, warmup=10000, thin=numc, init="random")
    post_normal_UIPJS = fit_UIPJS.extract() 
    result["UIPJS"] = post_normal_UIPJS

    print(fit_NPP)
    print(fit_LCP)
    print(fit_UIPD)
    print(fit_UIPJS)


    print(f"The iteration {jj+1}/{Num}."
        )
    results.append(result)


with open(f"MCMCNorm_{Num}_p0{int(100*p0)}_n{int(n)}.pkl", "wb") as f:
    pickle.dump(results, f)
