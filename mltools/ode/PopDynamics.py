import numpy as np
from .ODE import ODE

class PopDynamics(ODE):
    def __init__(self, r=1.0, K=1.0):
        super().__init__(r, K)

    def rhs(self, x, dt):
        return self.state()[0] * x * (1 - x /self.state()[1])