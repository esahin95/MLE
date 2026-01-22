import numpy as np
from .ODE import ODE 

class FreeFall(ODE):
    ''' 
    Free fall of point mass under constant gravity
    '''
    def __init__(self, g):
        self.g = g 
        
    def run(self, x, dt):
        t = 0
        while x[0] > 0:
            t += dt
            x += self.RK4(x, t, dt)
        
        # return falling time
        return t
        
    def rhs(self, x, t):
        y = np.roll(x, -1)
        y[-1] = -self.g
        return y