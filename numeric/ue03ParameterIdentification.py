# imports
import numpy as np
import matplotlib.pyplot as plt

from mltools.data import PopDynamics as Data
from mltools.ode import PopDynamics as ODE
from mltools.optim import NealderMead, makeFromOde

# ground truth
ds = Data(n=10, r=3.0, K=15.0, b=2.0)
T, X = ds.X, ds.y
plt.plot(T, X, 'ok')

# model objective
Objective = makeFromOde(ODE, T, X)
g = Objective(1e-1)

# parameter identification
opt = NealderMead()
x = opt(g, np.array([5.0, 5.0, 5.0]), eps=1e-8)

# test the solution
ode = ODE()
yPred = ode.run(x, 1e-2, ds.XTest)
plt.plot(ds.XTest, yPred, 'b')
plt.plot(ds.XTest, ds.yTest, '--r')
plt.show()