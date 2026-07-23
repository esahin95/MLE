from .DataCollection import DataCollection
from ..optim.Objectives import Monomial as F
import numpy as np

class Monomial(DataCollection):
    def __init__(self, *, fname='monomial', n=10, K=850, r=0.23, xMin=0.1, xMax=0.5):
        rng = np.random.default_rng(0)
        f = F(K=K, r=r)

        X = rng.uniform(xMin, xMax, size=(n,1))
        y = f(X)

        super().__init__(fname, 'tmp', X=X, y=y)