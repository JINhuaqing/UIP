from utils import ESSBetaCon, ESSBetaH
from scipy.stats import bernoulli
from numpy.random import uniform
import numpy as np
from tqdm import tqdm
import pickle


ns = [80, 100, 120]
n = 100
Ms = np.arange(10, 200, step=10)
ps_range = [0.4, 0.6]
pss = uniform(ps_range[0], ps_range[1], size=(50, 3))


ESSsM = []
for M in Ms:
    ESSs = []
    for ps in tqdm(pss):
        Ds = [ bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]
        ESS2 = ESSBetaH(ns=ns, Ds=Ds, n=n, M=M, C=1e6)
        ESS2.getESS()
        ESSs.append(ESS2.res)
    ESSsM.append(ESSs)
    

with open("./beta_ESS_UIPD_4_6.pkl", "wb") as f:
    pickle.dump(ESSsM, f)
