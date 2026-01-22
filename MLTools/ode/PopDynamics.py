import numpy as np
from .ODE import ODE 

class PopDynamics(ODE):
    def __init__(self):
        self._pi = np.array([1.0, 1.0])
    
    def rhs(self, x, dt):
        return self._pi[0] * x * (1 - x /self._pi[1])
    
    def run(self, pi, dt, T):
        # calculate the time stemps for solutioon
        n = len(T)
        tEnd = max(T)
        T = np.hstack((T, np.linspace(0, tEnd, np.floor(tEnd/dt).astype(np.int64))))
        T, I = np.unique(T, return_inverse=True)
        
        # calculate the solution vector
        self._pi = pi[:2]
        X = np.ones_like(T) * pi[2]
        for i in range(1, len(T)):
            X[i] = X[i-1] + self.RK4(X[i-1], T[i-1], T[i]-T[i-1])
            
        # return solution at input points
        return X[I[:n]]