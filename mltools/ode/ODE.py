import numpy as np
from ..utils import Parameter

class ODE(Parameter):
    def integrate(self, x, dt, *, tEnd=None, T=None):
        if (tEnd is None) == (T is None):
            raise ValueError('Either tEnd or T must be given')
        if T is None:
            n = np.floor(tEnd/dt).astype(np.int64)
            T = []
        if tEnd is None:
            n = T.size
            tEnd = np.max(T)

        T = np.hstack((
            T.flat,
            np.linspace(0, tEnd, np.floor(tEnd/dt).astype(np.int64))
        ))
        T, I = np.unique(T, return_inverse=True)

        X = np.tile(x, (T.size,1))
        for i in range(1, len(T)):
            X[i] = X[i-1] + self.RK4(X[i-1], T[i-1], T[i]-T[i-1])

        return X[I[:n]]

    def RK4(self, x, t, dt):
        k1 = self.rhs(x, t)
        k2 = self.rhs(x + 0.5*dt*k1, t + 0.5*dt)
        k3 = self.rhs(x + 0.5*dt*k2, t + 0.5*dt)
        k4 = self.rhs(x + dt*k3, t + dt)
        return dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0