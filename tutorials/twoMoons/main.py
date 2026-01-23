# imports
import sys
sys.path.append("..\\..\\")

import numpy as np
import matplotlib.pyplot as plt

from mltools.cluster import DBSCAN

# generate two moons data
nSamplesPerMoon = 240
noise = 0.02 
rng = np.random.default_rng(0)

T = np.linspace(0, np.pi, nSamplesPerMoon).reshape(-1,1)
X = np.block([[np.cos(T), np.sin(T)], [1-np.cos(T), 0.5-np.sin(T)]]);
X += rng.normal(loc=0.0, scale=noise, size=X.shape)
y = np.block([[np.zeros_like(T)], [np.ones_like(T)]])

fig, ax = plt.subplots(figsize=(4,3))
ax.scatter(X[:,0], X[:,1], c=y, s=5)
plt.show()

# train model
model = DBSCAN()
yPred = model.fit(X, 10, 0.08)

# postprocessing
fig, ax = plt.subplots(figsize=(4,3))
ax.scatter(X[:,0], X[:,1], c=yPred, s=5)
plt.show()      