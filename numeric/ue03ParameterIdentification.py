# imports
import numpy as np
import matplotlib.pyplot as plt

from mltools.data import PopDynamics as Data
from mltools.ode import PopDynamics as ODE
from mltools.optim import ODEObjective as Objective
from mltools.optim import NealderMead

# ground truth
ds = Data(n=10, r=3.0, K=15.0, b=2.0)
T, X = ds.X, ds.y
plt.plot(T.flat, X.flat, 'ok')

# model objective
g = Objective(ODE(), 1e-1, T, X)

# parameter identification
opt = NealderMead()
opt.fit(g, np.array([5.0, 5.0, 5.0]), maxIter=200, eps=1e-8)

# test the solution
s, x = g.split()
ode = ODE(*s)
yPred = ode.integrate(x, 1e-2, T=ds.XTest)
plt.plot(ds.XTest.flat, yPred.flat, 'b')
plt.plot(ds.XTest.flat, ds.yTest.flat, '--r')
plt.show()