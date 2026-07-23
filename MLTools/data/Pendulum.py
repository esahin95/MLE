from .DataCollection import DataCollection
from ..ode.Pendulum import Pendulum as ODE
import numpy as np

class Pendulum(DataCollection):
    def __init__(self, n=2000, sig=0.1, b=0.1, m=0.25, l=2.5, g=9.81):
        # differential equation solver
        ode = ODE(b, m, l, g)

        # generate dataset
        rng = np.random.default_rng(0)
        self.X = rng.uniform(-1, 1, size=(n,2)) * [np.pi, 10.0*np.sqrt(l/g)]
        self.y = np.array([ode.rhs(x, 0.)[1:2] for x in self.X]) + sig * rng.normal(size=(n, 1))