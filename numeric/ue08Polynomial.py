import numpy as np
import matplotlib.pyplot as plt

from mltools.optim import LogBarrier, LinearConstrained
from mltools.linear import Ridge

# ground truth
def f(X):
    return 3*X - X**2 + 2*X**3

# generate data
rng = np.random.default_rng(0)
m = 10
X = rng.uniform(-1, 1, size=(m,1))
y = f(X) + rng.normal(loc=0.0, scale=0.1, size=(m,1))

# feature map
p = 8
Phi = np.power(X[...,np.newaxis], np.arange(1,p+1)).reshape(X.shape[0],-1)

# objective
f = LinearConstrained(Phi, y, lam=0.1)

# constrained optimization
logB = LogBarrier(f)
logB.optimize(epochs=50, maxIter=50)

# unconstrained optimization
linR = Ridge(nFeatures=p)
linR.fit(Phi, y, lam=0.1)

# post processing
x = np.arange(p + 1).reshape(-1,1)
fig, ax = plt.subplots(1,1,figsize=(4,2))
ax.scatter(x, logB.x[:p+1], s=15, color='r', marker='o')
ax.scatter(x, linR.weights, s=15, color='b', marker='o')
plt.show()