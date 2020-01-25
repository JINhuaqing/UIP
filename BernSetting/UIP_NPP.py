from utilities_bern import *
import numpy as np
from scipy.stats import bernoulli
import scipy.stats as ss
import pymc3.stats as pymcs
import pickle
import argparse
import pystan
np.random.seed(0)

parser = argparse.ArgumentParser(description='UIP Bernoulli of NPP')
parser.add_argument('-p0', type=float, default=0.3, help='successful probability of current data')

args = parser.parse_args()

def load_pkl(f):
    with open(f, "rb") as fi:
        data = pickle.load(fi)
    return data


Num = 1000
p0 = args.p0
n = 80
fil = f"./betaMCMC1000n{n}nsdiff/MCMCBern_nsdiff_Num{Num}_p0{int(100*p0)}_n{int(n)}.pkl"
resdata =  load_pkl(fil)
results = []
print(f"The loading file is {fil}.")

NPPsm = pystan.StanModel(file="./Bernstan/bernNPP.stan")

for jj in range(Num):
    result = {}
    resdat = resdata[jj]["data"]
    D = resdat["D"]
    D1, D2 = resdat["Ds"]
    n = resdat["n"]
    n1, n2 = resdat["ns"]

    data = {"D":D, 
            "D1": D1,
            "D2": D2,
            "n" : n ,
            "n1": n1,
            "n2": n2
            }

    # NPP
    fit = NPPsm.sampling(data=data, 
        chains=4, 
        iter=30000, 
        warmup=5000,
        thin=10
        )
    print(fit)
    post_sps_npp = fit.extract()
    result["npp_sps"] = post_sps_npp



    print(f"The iteration {jj+1}/{Num}."
        )
    results.append(result)

with open(f"MCMCBernNPP_nsdiff_Num{Num}_p0{int(100*p0)}_n{int(n)}.pkl", "wb") as f:
    pickle.dump(results, f)
