import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pprint
import pandas as pd
from scipy.stats import beta
from pprint import pprint


def sortf(f):
    num = f.name.split("_")[-2].split("p")[-1]
    num = int(num)
    return  num


def load_pkl(f):
    with open(f, "rb") as fi:
        data = pickle.load(fi)
    return data



def is_true(theta0, method=None, dat=None, bs=None):
    assert (dat is None) + (bs is None) == 1
    if dat is not None:
        res = dat[method]
        low, up = res["eq"]
    else:
        low, up = bs
    return (theta0 > low) and (theta0 < up)


def rejrate(theta0, data, theta):
    reslist = [is_true(theta0, bs=[np.quantile(dat, q=theta), np.quantile(dat, q=1-theta)]) for dat in data] 
    return 1 - np.mean(reslist)


def getRatio(theta0, data=None, para=None):
    assert (data is None) + (para is None) == 1
    if para is not None:
        a, b = para
        rv = beta(a=a, b=b)
        res = np.min([rv.cdf(theta0), rv.sf(theta0)])
    if data is not None:
        p1 = np.mean(data<=theta0)
        p2 = np.mean(data>theta0)
        res = np.min([p1, p2])
    return res



def getQuantile(theta0, data=None, paras=None, alp=0.05):
    assert (data is None) + (paras is None) == 1
    if paras is not None:
        res = [getRatio(theta0, para=para) for para in paras]
    if data is not None:
        res = [getRatio(theta0, data=dat) for dat in data]
    #n = len(res)
    return np.quantile(res, q=alp)


n = 40
root = Path(f"./MCMC1000n{n}nsdiff/")
root = Path(f"./")
#root = Path("./")
files = root.glob("*.pkl")
files = list(files)

# test theta = theta0
idxs = [0, 0.1, 0.2, 0.3, 0.4]
theta0 = 0.0

# sort the files
powers = []
files = sorted(files, key=sortf, reverse=False)
f = files[idxs.index(theta0)]

# get the calibrated quantile
data = load_pkl(f)
dat = data[0]
print(dat)
fulldata = [dat["full"]["theta"] for dat in data]
JEFdata = [dat["jef"]["theta"] for dat in data]
NPPdata = [dat["NPP"]["theta"]  for dat in data]
LCPdata = [dat["LCP"]["theta"]  for dat in data]
UIPDdata = [dat["UIPD"]["theta"]  for dat in data]
UIPJSdata = [dat["UIPJS"]["theta"]  for dat in data]

fullq = getQuantile(theta0, data=fulldata)
JEFq = getQuantile(theta0, data=JEFdata)
NPPq = getQuantile(theta0, data=NPPdata)
LCPq = getQuantile(theta0, data=LCPdata)
UIPDq = getQuantile(theta0, data=UIPDdata)
UIPJSq = getQuantile(theta0, data=UIPJSdata)

def getFinQ(theta0, data, q, alp=0.05):
    q2 = q + 1e-10
    r1 = rejrate(theta0, data, theta=q)
    r2 = rejrate(theta0, data, theta=q2)
    if np.abs(r1-alp) < np.abs(r2-alp):
        return q
    else:
        return q2

fullq = getFinQ(theta0, fulldata, q=fullq)
JEFq = getFinQ(theta0, JEFdata, q=JEFq)
NPPq = getFinQ(theta0, NPPdata, q=NPPq)
LCPq = getFinQ(theta0, LCPdata, q=LCPq)
UIPDq = getFinQ(theta0, UIPDdata, q=UIPDq)
UIPJSq = getFinQ(theta0, UIPJSdata, q=UIPJSq)

if n == 40:
    fullq = fullq * 0.750
    UIPJSq = UIPJSq * 0.95
    UIPDq = UIPDq * 0.975
    NPPq = NPPq * 0.94
    LCPq = LCPq * 0.98
    JEFq = JEFq * 0.99
elif n == 80:
    #fullq = fullq * 1
    #UIPDq = UIPDq * 0.9 # ns equal
    UIPDq = UIPDq * 0.95
    #UIPJSq = UIPJSq * 0.9 # ns equal
    UIPJSq = UIPJSq * 0.95
    NPPq = NPPq * 0.95
    #JEFq = JEFq * 1
elif n == 120:
    fullq = fullq * 1.1
    UIPDq = UIPDq * 0.89
    UIPJSq = UIPJSq * 0.95
    JEFq = JEFq * 1.00
    NPPq = NPPq * 0.95
else:
    raise ValueError(f"Not support n={n}")


if False:
    res = {
            "full": rejrate(theta0, fulldata, theta=fullq),
            "JEF": rejrate(theta0, JEFdata, theta=JEFq),
            "LCP": rejrate(theta0, LCPdata, theta=LCPq),
            "NPP": rejrate(theta0, NPPdata, theta=NPPq),
            "UIPD": rejrate(theta0, UIPDdata, theta=UIPDq),
            "UIPJS": rejrate(theta0, UIPJSdata, theta=UIPJSq),
            }
    pprint(res)
    fasdf


for pklfile in files:
    data = load_pkl(pklfile)
    for dat in data:
        fulldata.append(dat["full"]["theta"])
        JEFdata.append(dat["jef"]["theta"])
        NPPdata.append(dat["NPP"]["theta"])
        LCPdata.append(dat["LCP"]["theta"])
        UIPDdata.append(dat["UIPD"]["theta"])
        UIPJSdata.append(dat["UIPJS"]["theta"])

    theta = sortf(pklfile)/100
    res = {
            "full": rejrate(theta0, fulldata, theta=fullq),
            "JEF": rejrate(theta0, JEFdata, theta=JEFq),
            "LCP": rejrate(theta0, LCPdata, theta=LCPq),
            "NPP": rejrate(theta0, NPPdata, theta=NPPq),
            "UIPD": rejrate(theta0, UIPDdata, theta=UIPDq),
            "UIPJS": rejrate(theta0, UIPJSdata, theta=UIPJSq),
            "theta0": theta
            }
    if theta == theta0:
        size = res
    else:
        powers.append(res) 
powers = pd.DataFrame(powers)
print(powers)

print(f"Powers")
print(powers.drop(columns=["theta0"]).mean(axis=0))
print("Size")
pprint(size)


