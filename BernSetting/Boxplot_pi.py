import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from BernSetting.utilities_bern import *
np.random.seed(1)


def GenD0(p0, n):
    num = int(n * p0)
    D0 = np.concatenate((np.ones(num), np.zeros(n-num)))
    return D0


p0s = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
n = 60
ps = [0.3, 0.8]
ns = [40, 40]
# D0s = [bernoulli.rvs(p0s[i], size=n) for i in range(len(p0s))]
D0s = [GenD0(p0s[i], n) for i in range(len(p0s))]
Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]

JSpis = {}
for D0, p0 in zip(D0s, p0s):
    JSpis[p0] = []
    calp, cbeta = 1 + np.sum(D0), 1 + np.sum(1-D0)
    for D in Ds:
        halp, hbeta = 1 + np.sum(D), 1 + np.sum(1-D)
        JSpis[p0].append(JS_dist_beta([calp, cbeta], [halp, hbeta]))

for key, v in JSpis.items():
    arrv = np.array(v)
    invv = 1/arrv
    pis = invv/np.sum(invv)
    JSpis[key] = pis




data = {}
pslist = []
histpslist = []
sps = []
for p0, D0 in zip(p0s, D0s):
    #post_sps_UIPD = gen_post_UIP_D(10000, D0, Ds, Maxiter=500)
    post_sps_UIPD = gen_post_UIP_D_MCMC(60000, D0, Ds, thin=50, burnin=10000)
    mulsps = post_sps_UIPD["sps_m"]
    dsps = mulsps/(mulsps.sum(axis=1).reshape(-1, 1))
    length = dsps.shape[0]
    print(f"p0 is {p0} and number of samples is {length}.")
    pslist = pslist + [p0] * length * 2
    histpslist = histpslist + [0.3] * length + [0.8] * length
    sps = sps + list(dsps[:, 0]) + list(dsps[:, 1])

dicdata = {"y": sps, "ps": pslist, "histps": histpslist}
dfdata = pd.DataFrame(dicdata)
sns.boxplot(y="y", x="ps", hue="histps", data=dfdata)
#plt.xticks(labels=p0s)
plt.xlabel(r"$\hat{\theta}$")
plt.ylabel(r"$\pi_k$")
plt.ylim([0, 1.3])
plt.legend(loc=1, title=r"$\theta_k$ of historical data")
plt.show()

JSpi1s = np.array([v[0] for v in JSpis.values()])
JSpi2s = np.array([v[1] for v in JSpis.values()])

plt.ylim([-0.1, 1.3])
plt.xlabel(r"$\hat{\theta}$")
plt.ylabel(r"$\pi_k$")
plt.xticks(p0s, labels=p0s)
plt.plot(p0s, JSpi1s, "bh", label="0.3")
plt.plot(p0s, JSpi2s, color="orange", marker="^", label="0.8", linestyle="")
plt.legend()
plt.show()
