import numpy as np
from .Objective import Objective


class ODEObjective(Objective):
    def __init__(self, ode, dt, T, X):
        super().__init__(n=len(ode)+X.shape[1])
        self.__ode = ode
        self.__T = T
        self.__X = X
        self.__dt = dt
        self.__n = len(ode)

    def split(self):
        return self.state()[:self.__n], self.state()[self.__n:]

    def eval(self, state=None):
        self.inc()
        if state is not None:
            self.set(state)
        s, x = self.split()

        self.__ode.set(s)
        X = self.__ode.integrate(x, self.__dt, T=self.__T)
        return np.sum((X - self.__X)**2)