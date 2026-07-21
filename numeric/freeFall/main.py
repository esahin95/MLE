import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

from mltools.linear import Linear
from mltools.ode import FreeFall

# seeded random number generator
rng = np.random.default_rng(0)
n = 20

# random initial conditions
v = rng.uniform(-0.5, 0.5, (n,1))
h = rng.uniform( 0.1, 2.0, (n,1))

# environment
g = 9.81

# integrate ODE
ode = FreeFall(g)
t = np.zeros((n,1))
for i in range(n):
    t[i] = ode.run(np.array([h[i], v[i]]), 1e-3)

# nondimensionalized input
X = v / np.sqrt(g * h)

# nondimensionalized output
y = t * np.sqrt(g / h)

# build model
model = Linear(nFeatures=1)
model.fit(X, y)
print(model)

# test on moon
g, h, v = 1.62, 1.5, 0.1
x = v / np.sqrt(h * g)
tTrue = np.sqrt(h / g) * (x + np.sqrt(x**2 + 2.0))
tPred = np.sqrt(h / g) * model(x.reshape(1,1))
print(tTrue, tPred)