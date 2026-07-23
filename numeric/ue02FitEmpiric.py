import numpy as np
import matplotlib.pyplot as plt

from mltools.data import DataCollection
from mltools.optim import makeFromFun, Monomial, GaussNewton

# load data
ds = DataCollection(fname="GaussNewton")

# model
Objective = makeFromFun(Monomial, ds.X, ds.y)
f = Objective()

# optimization
gs = GaussNewton()
gs.fit(f, eps=1e-10)
print(f.weights)

# post-processing
plt.scatter(ds.X, ds.y, color='b')
plt.scatter(ds.X, f(ds.X), facecolors='none', edgecolors='r')
plt.show()