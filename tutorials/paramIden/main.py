# imports
import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

from mltools.ode import PopDynamics
from mltools.utils import NealderMead

# establish ground truth
r, K, b = 3.0, 15.0, 2.0
xTrue = np.array([3.0, 15.0, 2.0])
T = np.linspace(0.5, 5.0, 10)
X = b / (b/K + (1 - b/K) * np.exp(-r*T)) 

# model objective
class Base:
    def __init__(self):
        self._nEval = 0

class G(Base):
    def __init__(self, ode, dt, T, X):
        super().__init__()
        self._ode = ode 
        self._dt = dt
        self._T = T 
        self._X = X
        
    def __call__(self, x):
        self._nEval += 1
        X = self._ode.run(x, self._dt, self._T)
        return np.sum((X - self._X)**2)
g = G(PopDynamics(), 1e-1, T, X)

# parameter identification
opt = NealderMead()
x = opt(g, np.array([5.0, 5.0, 5.0]), eps=1e-8)

# test the solution
ode = PopDynamics()
T = np.linspace(0.0, 5.0, 100)
X = ode.run(x, 1e-1, T)
Y = b / (b/K + (1 - b/K) * np.exp(-r*T)) 
plt.plot(T, X)
plt.plot(T, Y, '--')
plt.show()