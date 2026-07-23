import numpy as np
import matplotlib.pyplot as plt

from mltools.cluster import DBSCAN
from mltools.svm import SVCSE
from mltools import kwfig

# generate two moons data
nSamplesPerMoon = 340
noise = 0.05
rng = np.random.default_rng(0)

T = np.linspace(0, np.pi, nSamplesPerMoon).reshape(-1,1)
X = np.block([[np.cos(T), np.sin(T)], [1-np.cos(T), 0.5-np.sin(T)]]);
X += rng.normal(loc=0.0, scale=noise, size=X.shape)
y = np.block([[np.zeros_like(T)], [np.ones_like(T)]])

# limits
xLim = (np.floor(np.min(X[:,0])), np.ceil(np.max(X[:,0])))
yLim = (np.floor(np.min(X[:,1])), np.ceil(np.max(X[:,1])))
w, h = xLim[1]-xLim[0], yLim[1]-yLim[0]

# train supervised model
svm = SVCSE()
svm.fit(X, (y-0.5)*2, C=50, tol=1e-4, maxPasses=5)
Xt,Yt = np.meshgrid(np.linspace(xLim[0],xLim[1],200), np.linspace(yLim[0],yLim[1],200))
Pt = np.vstack((Xt.flat, Yt.flat)).T
Zt = (svm.predict(Pt).reshape(Xt.shape) + 1) / 2

# train unsupervised model
model = DBSCAN()
yPred = model.fit(X, 5, 0.08)

# Postprocessing
fig, axs = plt.subplots(1, 2, figsize=((w/h)*6, 3))
axs[0].contourf(Xt, Yt, Zt, cmap='copper', alpha=0.3, vmin=-1, vmax=1)
axs[0].scatter(X[:,0], X[:,1], c=y, s=5, cmap='copper', vmin=-1, vmax=1)
axs[0].axis('off')
axs[0].set_aspect('equal')
axs[1].scatter(X[:,0], X[:,1], c=yPred-1, s=5, cmap='copper', vmin=-1, vmax=1)
axs[1].axis('off')
axs[1].set_aspect('equal')
plt.show()