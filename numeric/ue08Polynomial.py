import numpy as np
import matplotlib.pyplot as plt

from mltools.optim import LogBarrier, LinearConstrained
from mltools.linear import Ridge
from mltools.data import Polynomial as Data

# generate data
ds = Data(coef=[2.0, -1.0, 3.0, 0.0])

# feature map
p = 8
Phi = np.power(ds.X[...,np.newaxis], np.arange(1,p+1)).reshape(ds.X.shape[0],-1)

# objective
f = LinearConstrained(Phi, ds.y, lam=0.1)

# constrained optimization
logB = LogBarrier(f)
logB.optimize(epochs=50, maxIter=50)

# unconstrained optimization
linR = Ridge(nFeatures=p)
linR.fit(Phi, ds.y, lam=0.1)

# post processing
x = np.arange(p + 1).reshape(-1,1)
fig, ax = plt.subplots(1,1,figsize=(4,2))
ax.scatter(x, logB.x[:p+1], s=15, color='r', marker='o')
ax.scatter(x, linR.weights, s=15, color='b', marker='o')
plt.show()