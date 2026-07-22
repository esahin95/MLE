import numpy as np

class ODE:
    def __init__(self, n):
        self.weights = np.zeros(n)

    def integrate(self, x, tEnd, dt, T=None, **kwargs):
        B = np.linspace(0, tEnd, np.floor(tEnd/dt).astype(np.int64))
        if T is None:
            n = B.size
            T = B
        else:
            n = T.size
            T = np.hstack((T,B))
        T, I = np.unique(T, return_inverse=True)

        X = np.zeros_like(T) + x
        for i in range(1, len(T)):
            X[i] = X[i-1] + self.RK4(X[i-1], T[i-1], T[i]-T[i-1])

        return X[I[:n]]


    def RK4(self, x, t, dt):
        k1 = self.rhs(x, t)
        k2 = self.rhs(x + 0.5*dt*k1, t + 0.5*dt)
        k3 = self.rhs(x + 0.5*dt*k2, t + 0.5*dt)
        k4 = self.rhs(x + dt*k3, t + dt)
        return dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0