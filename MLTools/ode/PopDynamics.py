import numpy as np
from .ODE import ODE

class PopDynamics(ODE):
    def __init__(self, r=1.0, K=1.0):
        super().__init__(2)
        self.weights[0] = r
        self.weights[1] = K

    def rhs(self, x, dt):
        return self.weights[0] * x * (1 - x /self.weights[1])


    def run(self, x, dt, T):
        self.weights.flat = x[:2]
        return self.integrate(x[2], max(T), dt, T)