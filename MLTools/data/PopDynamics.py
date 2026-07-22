from .DataCollection import DataCollection
from ..ode.PopDynamics import PopDynamics as ODE
import numpy as np

class PopDynamics(DataCollection):
    def __init__(self, n=10, b=2.0, **kwargs):
        self.X = np.arange(1, n+1) * 5.0 / n

        ode = ODE(**kwargs)
        self.y = ode.integrate(b, max(self.X), 1e-2, self.X)