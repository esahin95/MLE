import numpy as np

class ODE:
    def integrate(self, x, tStart, tEnd, dt):
        # number of time steps
        maxIt = np.ceil((tEnd - tStart) / dt).astype('int')
        
        # allocate storage
        X = np.zeros((maxIt, x.size))
        
        # time integration
        T = tStart + np.arange(maxIt) * dt
        for i, t in enumerate(T):
            X[i] = x
            x += self.RK4(x, t, dt)
            
        # return solution path
        return T, X

    def RK4(self, x, t, dt):
        k1 = self.rhs(x, t)
        k2 = self.rhs(x + 0.5*dt*k1, t + 0.5*dt)
        k3 = self.rhs(x + 0.5*dt*k2, t + 0.5*dt)
        k4 = self.rhs(x + dt*k3, t + dt)
        return dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

class FreeFall(ODE):
    ''' 
    Free fall of point mass under constant gravity
    '''
    def __init__(self, pi):
        self.pi = pi 
        
    def run(self, x, dt):
        t = 0
        while x[0] > 0:
            t += dt
            x += self.RK4(x, t, dt)
        
        # return falling time
        return t
        
    def rhs(self, x, t):
        y = np.roll(x, -1)
        y[-1] = -self.pi
        return y

class Pendulum(ODE):
    '''
    Pendulum class of ODE. Must implement the function rhs for RK4
    '''
    def __init__(self, b, m, l, g):
        self.c0 = -g/l
        self.c1 = -b/m

    def rhs(self, x, t):
        y = np.roll(x, -1)
        y[-1] = self.c0 * np.sin(x[0]) + self.c1 * x[1]
        return y