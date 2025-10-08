import numpy as np
import matplotlib.pyplot as plt

class ODE:
    '''
    General ODE solver with RK4
    '''
    def integrate(self, x, tStart, tEnd, dt):
        # number of time steps
        maxit = np.ceil((tEnd - tStart) / dt).astype('int')
        
        # allocate storage
        X = np.zeros((maxit, x.size))
        T = np.zeros((maxit, 1))
        
        # time integration
        T = np.arange(maxit) * (tEnd - tStart) + tStart
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

class Pendulum(ODE):
    '''
    Pendulum class of ODE. Must implement the function rhs for RK4
    '''
    def __init__(self, b, m, l, g):
        self.c0_ = -g/l
        self.c1_ = -b/m

    def rhs(self, x, t):
        y = x.copy()
        y[0] = x[1]
        y[1] = self.c0_ * np.sin(x[0]) + self.c1_ * x[1]
        return y
    

if __name__ == '__main__':
    # problem definition 
    b, m, l, g = 0.1, 0.25, 2.5, 9.81
    ode = Pendulum(b, m, l, g)
    
    # initial solution
    x = np.array([1.0, 1.0])
    
    # time solution
    ts, te, dt = 0., 10., 0.1
    T, X = ode.integrate(x, ts, te, dt)
    
    # plot angle
    plt.plot(T, X[:,0])
    plt.show()