import pystan
import pickle
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
np.random.seed(2020)


priortype = "UIPD"

theta0 = 0
n = 40
thetas = [0.5, 1]
ns = [50, 100]
sigma0 = sigma1 = sigma2 = 1

D0 = norm.rvs(loc=theta0, scale=sigma0, size=n)
D1 = norm.rvs(loc=thetas[0], scale=sigma1, size=ns[0])
D2 = norm.rvs(loc=thetas[1], scale=sigma2, size=ns[1])

data = {}
data['n'] = n
data['n1'] = ns[0]
data['n2'] = ns[1]
data['D'] = D0
data['D1'] = D1
data['D2'] = D2

control = {"adapt_delta": 0.90}

numc = 5
if priortype == "NPP":
    inits = [{"theta": 0, "sigma2": 1, "gamma1": 0.5, "gamma2": 0.5}] 
    sm = pystan.StanModel(file="./normNPP.stan")
elif priortype == "LCP":
    inits = [{"theta": 0, "sigma2": 1, "logtau1": 0, "logtau2": 0}] 
    sm = pystan.StanModel(file="./normLCP.stan")
elif priortype == "UIPD":
    inits = [{"theta": 0, "sigma2": 1, "M": 20, "pis": [0.5, 0.5]}] 
    sm = pystan.StanModel(file="./normUIPD.stan")
elif priortype == "UIPJS":
    inits = [{"theta": 0, "sigma2": 1, "M": 20}] 
    sm = pystan.StanModel(file="./normUIPJS.stan")

fit = sm.sampling(data=data, 
        chains=numc, 
        iter=40000, 
        warmup=20000,
        thin=numc,
        seed=2020,
        control=control,
#        algorithm="HMC",
        init=inits * numc
        )

print(fit)

dat = fit.extract()
plt.plot(dat["theta"][::10])
plt.savefig(f"./{priortype}.png")
#with open("stanJPP.pkl","wb") as f:
###with open("stanUIPD.pkl","wb") as f:
#    pickle.dump(dat, f)
