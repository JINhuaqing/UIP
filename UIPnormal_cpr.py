import numpy as np
from scipy.stats import norm
import pickle
import argparse
from utilities_norm import sigma2cond_theta, thetacond_sigma2

np.random.seed(2020)

parser = argparse.ArgumentParser(description='UIP Normal')
parser.add_argument('-t0', type=float, default=0.4, help='successful probability of current data')


Num = 1000
results = []
theta0 = args.t0
sigma0 = sigma1 = sigma2 = 1
n = 40
thetas = [1, 0.5]
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
