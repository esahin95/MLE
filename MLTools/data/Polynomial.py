from .DataCollection import DataCollection
from ..optim.Objectives import Polynomial as F
import numpy as np

class Polynomial(DataCollection):
    def __init__(self, *, fname='polynomial', n=10, coef=None, xMin=-1, xMax=1, scale=0.1):
        if coef is None:
            coef = [3.0, -1.0, 2.0]

        rng = np.random.default_rng(0)
        f = F(coef=coef)

        X = rng.uniform(xMin, xMax, size=(n,1))
        y = f(X) + rng.normal(loc=0.0, scale=scale, size=(n,1))

        super().__init__(fname, 'tmp', X=X, y=y)