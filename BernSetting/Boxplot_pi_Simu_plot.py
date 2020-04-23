import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict as ddict
import pandas as pd
import numpy as np

dataDir = Path("./")
figDir = Path("../plots")
if not figDir.exists():
    figDir.mkdir()
resFile = "Boxplot_pi_Simu100.pkl"
cleanFile = "Boxplot_pi_Simu100_cleaned.pkl"
with open(dataDir/resFile, "rb") as f:
    data = pickle.load(f)

if not (dataDir/cleanFile).exists():
    res = ddict(list)
    for dat in data:
        for p0 in dat:
            res[p0].append(dat[p0].mean(axis=0))
    for p0 in res:
        res[p0] = np.array(res[p0])
    with open(dataDir/cleanFile, "wb") as f:
        pickle.dump(res, f)
else:
    with open(dataDir/cleanFile, "rb") as f:
        res = pickle.load(f)

sps = []
psList = []
histpsList = []
for p0, dat in res.items():
    p0 = float(p0)
    length = dat.shape[0]
    psList = psList + [p0] * length * 2
    histpsList = histpsList + [r"$D_1$ $(\theta_1=$" + "0.3)"] * length + [r"$D_2$ $(\theta_2=$" + "0.8)"]* length
    sps = sps + list(dat[:, 0]) + list(dat[:, 1])
dicdata = {"y": sps, "ps": psList, "histps": histpsList}
dfdata = pd.DataFrame(dicdata)

sns.set_style("white")
sns.boxplot(y="y", x="ps", hue="histps", data=dfdata, palette=["#F25757", "#1094E9"])
#plt.xticks(labels=p0s)
plt.xlabel(r"$\theta$")
plt.ylabel(r"Posterior mean of $w_1/w_2$")
plt.ylim([0, 1.2])
plt.legend(loc=1, title=r"The historical datasets", frameon=True)
plt.savefig(figDir/"boxplot_pi_D_postm.pdf")
plt.show()
plt.close()
