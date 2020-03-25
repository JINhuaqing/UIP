import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from BernSetting.utilities_bern import *
import pickle
np.random.seed(1)


def GenD0(p0, n):
    num = int(n * p0)
    D0 = np.concatenate((np.ones(num), np.zeros(n-num)))
    return D0


p0s = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
n = 60
ps = [0.25, 0.40]
ns = [40, 40]
# D0s = [bernoulli.rvs(p0s[i], size=n) for i in range(len(p0s))]
D0s = [GenD0(p0s[i], n) for i in range(len(p0s))]
Ds = [GenD0(ps[i], ns[i]) for i in range(len(ns))]

pslist = []
sps = []
histpslist = []
for p0, D0 in zip(p0s, D0s):
    #post_sps_UIPD = gen_post_UIP_D(10000, D0, Ds, Maxiter=500)
    post_sps_UIPD = gen_post_UIP_D_MCMC(60000, D0, Ds, thin=50, burnin=10000)
    #post_sps_UIPJS = gen_post_UIP_KL(10000, D0, Ds, Maxiter=500)
    post_sps_UIPJS = gen_post_UIP_KL_MCMC(60000, D0, Ds, thin=50, burnin=10000)
    Dpoisps = post_sps_UIPD["sps_M"]
    JSpoisps = post_sps_UIPJS["sps_M"]
    length1 = Dpoisps.shape[0]
    length2 = JSpoisps.shape[0]
    print(f"p0 is {p0} and number of UIP-D samples is {length1}.")
    print(f"p0 is {p0} and number of UIP-JS samples is {length2}.")
    pslist = pslist + [p0] * (length1 + length2)
    sps = sps + list(Dpoisps) + list(JSpoisps)
    histpslist = histpslist + ["UIP-Dirichlet"] * length1 + ["UIP-JS"] * length2

dicdata = {"y": sps, "ps": pslist, "histps": histpslist}
dfdata = pd.DataFrame(dicdata)

with open("boxplot_M.pkl", "wb") as f:
    pickle.dump(dfdata, f)

with open("boxplot_M.pkl", "rb") as f:
    dfdata = pickle.load(f)

sns.boxplot(y="y", x="ps", hue="histps", data=dfdata)
# plt.xticks(labels=p0s)
plt.xlabel(r"$\hat{\theta}$")
plt.ylabel(r"$M$")
plt.ylim([0, 105])
plt.legend(loc=1, title="Priors")
plt.savefig("boxplot_M.pdf")
plt.show()
plt.close()
