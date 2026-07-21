import numpy as np
import matplotlib.pyplot as plt

from mltools.data import DataCollection
from mltools.utils import GaussNewton

# load data
ds = DataCollection(fname="GaussNewton")

# model
class F:
    def __init__(self, X, y):
        # model parameters
        self.weights = np.zeros((2,1))

        # data
        self._X = X
        self._y = y

    def eval(self):
        # residual
        h = self._X ** self.weights[1]
        f = self.weights[0] * h
        r = self._y - f

        # derivatives
        gradf = np.hstack((h, f*np.log(ds.X)))

        # return tuple
        return r, gradf

    def __call__(self, X):
        return self.weights[0] * X ** self.weights[1]
f = F(ds.X, ds.y)

# optimization
gs = GaussNewton()
gs.fit(f, eps=1e-10)
print(f.weights)

# post-processing
plt.scatter(ds.X, ds.y, color='b')
plt.scatter(ds.X, f(ds.X), facecolors='none', edgecolors='r')
plt.show()