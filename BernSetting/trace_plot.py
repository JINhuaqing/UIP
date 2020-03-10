import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from utilities_bern import *
np.random.seed(1)


def GenD0(p0, n):
    num = int(n * p0)
    D0 = np.concatenate((np.ones(num), np.zeros(n-num)))
    return D0


p0 =0.3
n = 60
ps = [0.3, 0.8]
ns = [40, 40]
D0 = GenD0(p0, n)
Ds = [GenD0(ps[i], ns[i]) for i in range(len(ns))]
#Ds = [bernoulli.rvs(ps[i], size=ns[i]) for i in range(len(ns))]

#post_sps_UIPD = gen_post_UIP_D(10000, D0, Ds, Maxiter=500)
post_sps_UIPD = gen_post_UIP_D_MCMC(10000, D0, Ds, thin=1, burnin=5000, diag=True)
# post_sps_UIPJS = gen_post_UIP_KL(10000, D0, Ds, Maxiter=500)
post_sps_UIPJS = gen_post_UIP_KL_MCMC(10000, D0, Ds, thin=1, burnin=5000, diag=True)

# plot of UIP-JS prior
plt.subplot(211)
plt.title("Posterior traceplot under UIP-JS prior")
plt.plot(post_sps_UIPJS["sps"], color="red")
plt.xlabel("T")
plt.ylabel(r"$\theta_{post}$")
plt.subplot(212)
plt.plot(post_sps_UIPJS["sps_M"], color="green")
plt.xlabel("T")
plt.ylabel(r"$M_{post}$")
plt.savefig("UIPJS_traceplot.jpg")
plt.show()


# plot of UIP-D prior
plt.subplot(211)
plt.title("Posterior traceplot under UIP-D prior")
plt.plot(post_sps_UIPD["sps"], color="red")
plt.xlabel("T")
plt.ylabel(r"$\theta_{post}$")
plt.subplot(212)
plt.plot(post_sps_UIPD["sps_M"], color="green")
plt.xlabel("T")
plt.ylabel(r"$M_{post}$")
plt.savefig("UIPD_traceplot.jpg")
plt.show()


# posterior density
post_sps_UIPD_RJ = gen_post_UIP_D(10000, D0, Ds, Maxiter=50)
post_sps_UIPD_MCMC = gen_post_UIP_D_MCMC(50000, D0, Ds, thin=10, burnin=5000)
post_sps_UIPJS_RJ = gen_post_UIP_KL(10000, D0, Ds, Maxiter=50)
post_sps_UIPJS_MCMC = gen_post_UIP_KL_MCMC(50000, D0, Ds, thin=10, burnin=5000)


#plt.suptitle("Posterior densities under rejection sampling and MH sampling", verticalalignment='bottom')
plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.subplot(221)
sns.kdeplot(post_sps_UIPD_RJ["sps"], color="red", label="RJ Sampling")
sns.kdeplot(post_sps_UIPD_MCMC["sps"], color="green", label="MH Sampling")
#sns.kdeplot(post_sps_jpp["sps"].reshape(-1), color="blue", label="JPP Sampling")
#sns.kdeplot(data1["theta"], color="pink", label="PP prior")
plt.ylabel("")
plt.xlabel(r"$\theta_{post}$")
plt.legend(loc=1)

plt.subplot(222)
sns.kdeplot(post_sps_UIPD_RJ["sps_M"], color="red", label="RJ Sampling")
sns.kdeplot(post_sps_UIPD_MCMC["sps_M"], color="green", label="MH Sampling")
#sns.kdeplot(data["M"], color="blue", label="Stan Sampling")
plt.ylabel("")
plt.xlabel(r"$M_{post}$")
plt.legend(loc=1)

plt.subplot(223)
sns.kdeplot(post_sps_UIPJS_RJ["sps"], color="red", label="RJ Sampling")
sns.kdeplot(post_sps_UIPJS_MCMC["sps"], color="green", label="MH Sampling")
plt.ylabel("")
plt.xlabel(r"$\theta_{post}$")
plt.legend(loc=1)

plt.subplot(224)
sns.kdeplot(post_sps_UIPJS_RJ["sps_M"], color="red", label="RJ Sampling")
sns.kdeplot(post_sps_UIPJS_MCMC["sps_M"], color="green", label="MH Sampling")
plt.ylabel("")
plt.yticks(np.arange(0, 0.030, 0.005))
plt.xlabel(r"$M_{post}$")
plt.legend(loc=1)

plt.savefig("cpr_rj_MH.jpg")
plt.show()