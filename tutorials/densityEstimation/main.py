# imports
import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from mltools.cluster import GMM 

# ground truth
rng = np.random.default_rng(0)
k = 3
d = 2
ns = rng.choice(100, size=(k))
phi = ns / np.sum(ns)
mus = rng.normal(size=(k,d)) * 4
sigmas = rng.normal(size=(k,d,d))
for i in range(sigmas.shape[0]):
    sigmas[i] = sigmas[i] @ sigmas[i].T + 0.1 * np.eye(d)
    
def prob(X):
    m, d = X.shape
    
    W = np.zeros((m, k))
    for j in range(k):
        p = multivariate_normal(mean=mus[j], cov=sigmas[j])
        W[:,j] = p.pdf(X)
    return np.sum(W * phi, axis=1, keepdims=True)

# generate data
X = np.zeros((np.sum(ns), d))
nc = 0
for i, n in enumerate(ns):
    X[nc:nc+n] = rng.multivariate_normal(mus[i], sigmas[i], size=ns[i])
    nc += n
y = np.repeat(range(k), ns).reshape(-1,1)
p = rng.permutation(X.shape[0])
X = X[p]
y = y[p]

# train model
model = GMM(X, k)
model.fit(maxIter=18)

# test points
Xm, Ym = np.meshgrid(
    np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100),
    np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
)
Pm = np.hstack((Xm.reshape(-1,1), Ym.reshape(-1,1)))

# compare probability density
fig, axs = plt.subplots(1,2,figsize=(10,4))
Zm = prob(Pm).reshape(Xm.shape)
axs[0].contourf(Xm, Ym, Zm)
axs[0].scatter(X[:,0], X[:,1], c='k')
Zm = model.predict(Pm).reshape(Xm.shape)
axs[1].contourf(Xm, Ym, Zm)
axs[1].scatter(X[:,0], X[:,1], c='k')
plt.show()