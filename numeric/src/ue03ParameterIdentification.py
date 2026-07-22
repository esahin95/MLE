# imports
import numpy as np
import matplotlib.pyplot as plt

from mltools.data import PopDynamics as Data
from mltools.ode import PopDynamics as ODE
from mltools.utils import NealderMead, makeFromOde

# establish ground truth
r, K, b = 3.0, 15.0, 2.0
ds = Data(n=10, r=r, K=K, b=b)
T, X = ds.X, ds.y

#T = np.linspace(0.5, 5.0, 10)
#X = b / (b/K + (1 - b/K) * np.exp(-r*T))
plt.plot(T,X,'ok')

# model objective
Objective = makeFromOde(ODE, T, X)
g = Objective(1e-1)

# parameter identification
opt = NealderMead()
x = opt(g, np.array([5.0, 5.0, 5.0]), eps=1e-8)

# test the solution
ode = ODE()
T = np.linspace(0.0, 5.0, 100)
X = ode.run(x, 1e-1, T)
Y = b / (b/K + (1 - b/K) * np.exp(-r*T))
plt.plot(T, X, 'b')
plt.plot(T, Y, '--r')
plt.show()