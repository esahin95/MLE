from .DataCollection import DataCollection
from ..ode import Pendulum as ODE
import numpy as np

class Pendulum(DataCollection):
    def __init__(self, *, fname='pendulum', n=2000, sig=0.1, b=0.1, m=0.25, l=2.5, g=9.81, fac=10.0):
        # differential equation solver
        ode = ODE(b, m, l, g)

        # generate dataset
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(n,2)) * [np.pi, fac*np.sqrt(l/g)]
        y = np.array([ode.rhs(x, 0.)[1:2] for x in X]) + sig * rng.normal(size=(n, 1))

        super().__init__(fname, 'tmp', X=X, y=y)