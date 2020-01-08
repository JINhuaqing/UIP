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
    num = f.name.split("_")[2].split("p")[-1]
    num = int(num)
    return  num


def load_pkl(f):
    with open(f, "rb") as fi:
        data = pickle.load(fi)
    return data


def is_valid(dat, num=1000):
    numsps1 = len(dat["UIPm_sps"]["sps"])
    numsps2 = len(dat["UIPKL_sps"]["sps"])
    return (numsps1 > num) and (numsps2 > num)


def is_true(p0, method=None, dat=None, bs=None):
    assert (dat is None) + (bs is None) == 1
    if dat is not None:
        res = dat[method]
        low, up = res["eq"]
    else:
        low, up = bs
    return (p0 > low) and (p0 < up)


def rejrates(p0, data):
    ress = {"jef":[], "full":[], "jpp":[], "UIPKL":[], "UIPm":[]}
    for dat in data:
        for method in ress.keys():
            ress[method].append(is_true(p0, dat, method))
    for res in ress.items():
        keyv, reslist = res
        ress[keyv] = 1 - np.mean(reslist)
    return ress

def rejrate(p0, data, q):
    if len(data[0]) > 2:
        reslist = [is_true(p0, bs=[np.quantile(dat, q=q), np.quantile(dat, q=1-q)]) for dat in data] 
    else:
        reslist = [is_true(p0, bs=[beta.ppf(q=q, a=dat[0], b=dat[1]), beta.ppf(q=1-q, a=dat[0], b=dat[1])]) for dat in data] 
    return 1 - np.mean(reslist)


def getRatio(p0, data=None, para=None):
    assert (data is None) + (para is None) == 1
    if para is not None:
        a, b = para
        rv = beta(a=a, b=b)
        res = np.min([rv.cdf(p0), rv.sf(p0)])
    if data is not None:
        p1 = np.mean(data<=p0)
        p2 = np.mean(data>p0)
        res = np.min([p1, p2])
    return res



def getQuantile(p0, data=None, paras=None, alp=0.05):
    assert (data is None) + (paras is None) == 1
    if paras is not None:
        res = [getRatio(p0, para=para) for para in paras]
    if data is not None:
        res = [getRatio(p0, data=dat) for dat in data]
    #n = len(res)
    return np.quantile(res, q=alp)


root = Path("./betaMCMC1000/")
root = Path("./")
files = root.glob("*.pkl")
files = list(files)

# test p = p0
idxs = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
p0 = 0.50

# sort the files
powers = []
files = sorted(files, key=sortf, reverse=False)
f = files[idxs.index(p0)]

# get the calibrated quantile
data = load_pkl(f)
fulldata = [dat["full_popu"] for dat in data]
JEFdata = [dat["jef_popu"] for dat in data]
JPPdata = [dat["jpp_sps"]["sps"]  for dat in data]
UIPDdata = [dat["UIPD_sps"]["sps"]  for dat in data]
UIPKLdata = [dat["UIPKL_sps"]["sps"]  for dat in data]

fullq = getQuantile(p0, paras=fulldata)
JEFq = getQuantile(p0, paras=JEFdata)
JPPq = getQuantile(p0, data=JPPdata)
UIPDq = getQuantile(p0, data=UIPDdata)
UIPKLq = getQuantile(p0, data=UIPKLdata)

def getFinQ(p0, data, q, alp=0.05):
    q2 = q + 1e-10
    r1 = rejrate(p0, data, q=q)
    r2 = rejrate(p0, data, q=q2)
    if np.abs(r1-alp) < np.abs(r2-alp):
        return q
    else:
        return q2

fullq = getFinQ(p0, fulldata, q=fullq)
JEFq = getFinQ(p0, JEFdata, q=JEFq)
JPPq = getFinQ(p0, JPPdata, q=JPPq)
UIPDq = getFinQ(p0, UIPDdata, q=UIPDq)
UIPKLq = getFinQ(p0, UIPKLdata, q=UIPKLq)
#JEFq = JEFq * 1.01

print(fullq, JEFq, JPPq, UIPDq, UIPKLq)



for pklfile in files:
    data = load_pkl(pklfile)
    data = [dat for dat in data if len(dat["UIPKL"]) != 0 and len(dat["UIPD"]) != 0]
    JEFdata = [dat["jef_popu"] for dat in data]
    fulldata = [dat["full_popu"] for dat in data]
    UIPDdata = [dat["UIPD_sps"]["sps"]  for dat in data]
    UIPKLdata = [dat["UIPKL_sps"]["sps"]  for dat in data]
    JPPdata = [dat["jpp_sps"]["sps"]  for dat in data]

    p = sortf(pklfile)/100
    res = {
            "full": rejrate(p0, fulldata, q=fullq),
            "JEF": rejrate(p0, JEFdata, q=JEFq),
            "JPP": rejrate(p0, JPPdata, q=JPPq),
            "UIPD": rejrate(p0, UIPDdata, q=UIPDq),
            "UIPKL": rejrate(p0, UIPKLdata, q=UIPKLq),
            "p0": p
            }
    if p == p0:
        size = res
    else:
        powers.append(res) 
powers = pd.DataFrame(powers)
print(powers)

print(f"Powers")
print(powers.drop(columns=["p0"]).mean(axis=0))
print("Size")
pprint(size)


