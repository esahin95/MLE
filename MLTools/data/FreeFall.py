from .DataCollection import DataCollection
from ..ode.FreeFall import FreeFall as ODE
import numpy as np

class FreeFall(DataCollection):
    def __init__(self, *args, n=20, **kwargs):
        rng = np.random.default_rng(0)
        n = 20

        # random initial conditions
        v = rng.uniform(-0.5, 0.5, (n,1))
        h = rng.uniform( 0.1, 2.0, (n,1))

        # environment
        g = 9.81

        # integrate ODE
        ode = ODE(g)
        t = np.zeros((n,1))
        for i in range(n):
            t[i] = ode.run(np.array([h[i], v[i]]), 1e-3)

        # nondimensionalized data
        self.X = v / np.sqrt(g * h)
        self.y = t * np.sqrt(g / h)