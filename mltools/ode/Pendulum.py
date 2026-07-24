import numpy as np
from .ODE import ODE

class Pendulum(ODE):
    '''
    Pendulum class of ODE. Must implement the function rhs for RK4
    '''
    def __init__(self, b, m, l, g):
        super().__init__(2)
        self.weights[0] = -g/l
        self.weights[1] = -b/m

    def rhs(self, x, t):
        y = np.roll(x, -1)
        y[-1] = self.weights[0] * np.sin(x[0]) + self.weights[1] * x[1]
        return y