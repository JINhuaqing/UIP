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
resFiles = dataDir.glob("Boxplot_M*[05].pkl")
resFiles = sorted(list(resFiles))
cleanFile = "Boxplot_M_Simu100_cleaned.pkl"
with open(dataDir/resFiles[0], "rb") as f:
    dataD = pickle.load(f)
keys = list(dataD[0].keys())
if not (dataDir/cleanFile).exists():
    resD = ddict(list)
    for datD in dataD:
        for p0 in datD:
            #resD[p0].append(np.median(datD[p0]))
            resD[p0].append(datD[p0].mean())

    resJS = ddict(list)
    for idx, resFile in enumerate(resFiles[1:]):
        with open(dataDir/resFile, "rb") as f:
            dataJS = pickle.load(f)
        dataJS = np.array(dataJS)
#        resJS[keys[idx]] = list(np.median(dataJS, axis=1))
        resJS[keys[idx]] = list(dataJS.mean(axis=1))
    with open(dataDir/cleanFile, "wb") as f:
        pickle.dump([resD, resJS], f)
else:
    with open(dataDir/cleanFile, "rb") as f:
        resD, resJS = pickle.load(f)

sps = []
psList = []
histpsList = []
for p0 in keys:
    curD = resD[p0]
    curJS = resJS[p0]
    lenD, lenJS = len(curD), len(curJS)
    p0 = float(p0)
    psList = psList + [p0] * (lenD+lenJS)
    sps = sps + list(curD) + list(curJS)
    histpsList = histpsList + ["UIP-Dirichlet"] * lenD + ["UIP-JS"] * lenJS

dicdata = {"y": sps, "ps": psList, "histps": histpsList}
dfdata = pd.DataFrame(dicdata)

sns.set_style("white")
sns.boxplot(y="y", x="ps", hue="histps", data=dfdata, palette="Set3")
# plt.xticks(labels=p0s)
plt.xlabel(r"$\theta$")
plt.ylabel(r"Posterior mean of $M$")
plt.ylim([0, 60])
plt.legend(loc=1, title="Priors", frameon=True)
plt.savefig(figDir/"boxplot_M_postm.pdf")
plt.show()
plt.close()
