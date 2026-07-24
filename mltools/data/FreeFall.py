from .DataCollection import DataCollection
from ..ode import FreeFall as ODE
import numpy as np

class FreeFall(DataCollection):
    def __init__(self, *, fname='freeFall', n=20, g=9.81, vmin=-0.5, vmax=0.5, hmin=0.1, hmax=2.0, dt=1e-3):
        rng = np.random.default_rng(0)

        # random initial conditions
        v = rng.uniform(vmin, vmax, (n,1))
        h = rng.uniform(hmin, hmax, (n,1))

        # integrate ODE
        ode = ODE(g)
        t = np.zeros((n,1))
        for i in range(n):
            t[i] = ode.run(np.array([h[i], v[i]]), dt)

        # nondimensionalized data
        X = v / np.sqrt(g * h)
        y = t * np.sqrt(g / h)

        super().__init__(fname, 'tmp', X=X, y=y)