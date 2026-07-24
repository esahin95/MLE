import numpy as np
import matplotlib.pyplot as plt

from mltools.data import Monomial as Data
from mltools.optim import Monomial as Objective
from mltools.optim import GaussNewton

# load data
ds = Data()

# model
f = Objective(ds.X, ds.y)
print(f.state())

# optimization
gs = GaussNewton()
gs.fit(f, eps=1e-10)
print(f.state())

# post-processing
plt.scatter(ds.X, ds.y, color='b')
plt.scatter(ds.X, f(ds.X), facecolors='none', edgecolors='r')
plt.show()