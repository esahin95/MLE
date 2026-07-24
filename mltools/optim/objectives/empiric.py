import numpy as np
from .Objective import Objective


class Base(Objective):
    def __init__(self, X, y, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.__X = X
        self.__y = y

    def df(self, X):
        raise NotImplementedError()

    def __call__(self, X):
        raise NotImplementedError()

    def eval(self):
        self.inc()
        return self.__y - self(self.__X), self.df(self.__X), None


class Monomial(Base):
    def __init__(self, X=None, y=None, *, K=0.0, r=0.0):
        super().__init__(X, y, K, r)

    def df(self, X):
        return np.hstack((
            X ** self.state()[1],
            self.state()[0] * np.log(X) * X ** self.state()[1]
        ))

    def __call__(self, X):
        return self.state()[0] * X ** self.state()[1]


class Polynomial(Base):
    def __init__(self, X=None, y=None, *, coef=(3.0, -1.0, 2.0, 0.0)):
        super().__init__(X, y, *coef)

    def df(self, X):
        return np.power(X, np.arange(4)[::-1])

    def __call__(self, X):
        y = self.state()[0]
        for w in self.state()[1:]:
            y = y * X + w
        return y