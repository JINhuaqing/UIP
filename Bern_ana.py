import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


root = Path("./")
files = root.glob("*.pkl")
files = list(files)


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


def is_true(p0, dat, method):
    res = dat[method]
    low, up = res["eq"]
    return (p0 > low) and (p0 < up)


def rejrate(p0, data):
    ress = {"jef":[], "full":[], "jpp":[], "UIPKL":[], "UIPm":[]}
    for dat in data:
        for method in ress.keys():
            ress[method].append(is_true(p0, dat, method))
    for res in ress.items():
        keyv, reslist = res
        ress[keyv] = 1 - np.mean(reslist)
    return ress


# sort the files
files = sorted(files, key=sortf, reverse=False)
for pklfile in files:
    data = load_pkl(pklfile)
    data = [dat for dat in data if is_valid(dat)]
    p = sortf(pklfile)/100
    print(p, rejrate(0.3, data))


