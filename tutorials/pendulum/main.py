# imports
import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

from mltools.ode import Pendulum
from mltools.linear import Ridge

# pendulum
b, m, l, g = 0.1, 0.25, 2.5, 9.81
ode = Pendulum(b, m, l, g)

# generate dataset
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(2000,2)) * [np.pi, 10.0*np.sqrt(l/g)]
Y = np.array([ode.rhs(x, 0.)[1:2] for x in X]) + 0.1 * rng.normal(size=(X.shape[0], 1))
X = np.hstack((X, np.sin(X), np.cos(X)))
X = np.power(X[...,np.newaxis],np.arange(1,4)).reshape(X.shape[0],-1)

# build model
model = Ridge(X.shape[-1])
model.fit(X[:1000], Y[:1000], lam=1.)

# post process
YPred = model(X[1000:])
plt.scatter(Y[1000:],YPred)
plt.show()