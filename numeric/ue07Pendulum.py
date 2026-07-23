import numpy as np
import matplotlib.pyplot as plt

from mltools.data import Pendulum
from mltools.linear import Ridge

# pendulum
ds = Pendulum(b=0.1, m=0.25, l=2.5, g=9.81)

# feature mapping
X = np.hstack((ds.X, np.sin(ds.X), np.cos(ds.X)))
X = np.power(X[...,np.newaxis],np.arange(1,4)).reshape(X.shape[0],-1)

# build model
model = Ridge(X.shape[-1])
model.fit(X[:1000], ds.y[:1000], lam=0.1)
print(model)

# post process
YPred = model(X[1000:])
plt.scatter(ds.y[1000:],YPred)
plt.show()