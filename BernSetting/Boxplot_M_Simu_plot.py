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

spsD = []
spsJS = []
psListD = []
psListJS = []
histpsListD = []
histpsListJS = []
for p0 in keys:
    curD = resD[p0]
    curJS = resJS[p0]
    lenD, lenJS = len(curD), len(curJS)
    p0 = float(p0)
    psListD = psListD + [p0] * lenD
    psListJS = psListJS + [p0] * lenJS
    spsD = spsD + list(curD)
    spsJS = spsJS + list(curJS)

dicdataD = {"y": spsD, "ps": psListD}
dicdataJS = {"y": spsJS, "ps": psListJS}
dfdataD = pd.DataFrame(dicdataD)
dfdataJS = pd.DataFrame(dicdataJS)

sns.set_style("white")
sns.boxplot(y="y", x="ps", data=dfdataD, palette=["#F25757"])
plt.xlabel(r"$\theta$")
plt.ylabel(r"Posterior mean of $M$")
plt.ylim([0, 50])
plt.savefig(figDir/"boxplot_M_D_postm.pdf")
plt.show()
plt.close()

sns.set_style("white")
sns.boxplot(y="y", x="ps", data=dfdataJS, palette=["#F25757"])
plt.xlabel(r"$\theta$")
plt.ylabel(r"Posterior mean of $M$")
plt.ylim([0, 50])
plt.savefig(figDir/"boxplot_M_JS_postm.pdf")
plt.show()
plt.close()
