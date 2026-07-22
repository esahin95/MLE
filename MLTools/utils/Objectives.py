import numpy as np

def makeFromFun(Base, X, y, *args, **kwargs):
    class Derived(Base):
        def __init__(self):
            super().__init__(*args, **kwargs)

            self.X = X
            self.y = y

        def eval(self):
            f, df = super().eval(self.X)
            return self.y - f, df

    return Derived

def makeFromOde(Base, X, y, *args, **kwargs):
    class Derived(Base):
        def __init__(self, dt):
            super().__init__(*args, **kwargs)

            self.nEval = 0
            self.dt = dt

            self.T = X
            self.X = y

        def eval(self):
            raise NotImplementedError()

        def __call__(self, x):
            self.nEval += 1
            X = self.run(x, self.dt, self.T)
            return np.sum((X - self.X)**2)

    return Derived


class Monomial:
    def __init__(self):
        self.weights = np.zeros((2,1))
        self.nEval = 0

    def eval(self, X):
        df = np.hstack((
            X ** self.weights[1],
            self.weights[0] * np.log(X) * X ** self.weights[1]
        ))

        return self(X), df

    def __call__(self, X):
        self.nEval += 1
        return self.weights[0] * X ** self.weights[1]