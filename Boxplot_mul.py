import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from utilities_bern import *

np.random.seed(1)
p0s = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
p0s = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
n = 50
ps = [0.3, 0.8]
ns = [40, 40]
D0s = [bernoulli.rvs(p0s[i], size=n) for i in range(len(p0s))]
Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]

data = {}
pslist = []
histpslist = []
sps = []
for p0, D0 in zip(p0s, D0s):
    post_sps_UIPm = gen_post_UIP_multi(10000, D0, Ds, Maxiter=500)
    mulsps = post_sps_UIPm["sps_mul"]
    length = mulsps.shape[0]
    print(f"p0 is {p0} and number of samples is {length}.")
    pslist = pslist + [p0] * length * 2
    histpslist = histpslist + [0.3] * length + [0.8] * length
    sps = sps + list(mulsps[:, 0]) + list(mulsps[:, 1])

dicdata = {"y": sps, "ps": pslist, "histps": histpslist}
dfdata = pd.DataFrame(dicdata)
sns.boxplot(y="y", x="ps", hue="histps", data=dfdata)
#plt.xticks(labels=p0s)
plt.xlabel("p0")
plt.ylabel(r"$M_i$")
plt.ylim([0, 35])
plt.legend(loc=1, title="p of historical data")
plt.show()

