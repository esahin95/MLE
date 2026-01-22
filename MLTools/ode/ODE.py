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