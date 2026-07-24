import numpy as np
from .ODE import ODE

class Pendulum(ODE):
    '''
    Pendulum class of ODE. Must implement the function rhs for RK4
    '''
    def __init__(self, b, m, l, g):
        super().__init__(-g/l, -b/m)

    def rhs(self, x, t):
        y = np.roll(x, -1)
        y[-1] = self.state()[0] * np.sin(x[0]) + self.state()[1] * x[1]
        return y