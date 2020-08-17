import numpy as np
from collections import defaultdict as ddict
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from utilities_bern import *
import pickle
from tqdm import tqdm
np.random.seed(1)


p0s = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
n = 60
ps = [0.3, 0.8]
ns = [40, 40]
numRep = 100
saveDir = Path("./") 
figDir = Path("../plots")
saveName =f"Boxplot_pi_JS_Simu{numRep}.pkl" 

if not (saveDir/saveName).exists():
    Allres = ddict(list)
    for idx in tqdm(range(1, numRep+1)):
        D0s = [bernoulli.rvs(p0s[i], size=n) for i in range(len(p0s))]
        Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]
    
        for p0, D0 in zip(p0s, D0s):
            calp, cbeta = 1 + np.sum(D0), 1 + np.sum(1-D0)
            oneRes = []
            for D in Ds:
                halp, hbeta = 1 + np.sum(D), 1 + np.sum(1-D)
                oneRes.append(JS_dist_beta([calp, cbeta], [halp, hbeta]))
            oneResArr = np.array(oneRes)
            invOneResArr = 1/oneResArr
            pis = invOneResArr/np.sum(invOneResArr)
            Allres[f"{p0}"].append(pis)
    
    with open(saveDir/saveName, "wb") as f:
        pickle.dump(Allres, f)
else:
    with open(saveDir/saveName, "rb") as f:
        Allres = pickle.load(f)

psList = []
histpsList = []
sps = []
for p0 in p0s:
    curRes = Allres[f"{p0}"]
    curRes = np.array(curRes)
    sps = sps + list(curRes[:, 0]) + list(curRes[:, 1])
    psList = psList + [p0]*curRes.size
    histpsList = histpsList + [r"$D_1$ $(\theta_1=$" + "0.3)"] * len(curRes)+ [r"$D_2$ $(\theta_2=$" + "0.8)"]*len(curRes)
    
dicdata = {"y": sps, "ps": psList, "histps": histpsList}
dfdata = pd.DataFrame(dicdata)

sns.set_style("white")
sns.boxplot(y="y", x="ps", hue="histps", data=dfdata, palette=["#F25757", "#1094E9"])
#plt.xticks(labels=p0s)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$w_1/w_2$")
plt.ylim([0, 1.33])
plt.legend(loc=1, title=r"Historical datasets", frameon=True)
plt.savefig(figDir/"boxplot_pi_JS.pdf")
plt.show()
plt.close()


    
