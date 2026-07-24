import numpy as np

class Objective:
    def __init__(self, n):
        self.weights = np.zeros(n)
        self.nEval = 0

    def __call__(self, X):
        raise NotImplementedError()

    def eval(self, X):
        raise NotImplementedError()

    def ineq(self, X):
        raise NotImplementedError()

    def __len__(self):
        return self.weights.size

    def inc(self):
        self.nEval += 1


class Monomial(Objective):
    def __init__(self, *, K=0.0, r=0.0):
        super().__init__(2)
        self.weights[0] = K
        self.weights[1] = r

    def eval(self, X):
        df = np.hstack((
            X ** self.weights[1],
            self.weights[0] * np.log(X) * X ** self.weights[1]
        ))

        return self(X), df, None

    def __call__(self, X):
        self.inc()
        return self.weights[0] * X ** self.weights[1]


class Polynomial(Objective):
    def __init__(self, *, coef):
        super().__init__(len(coef))
        for i in range(len(coef)):
            self.weights[i] = coef[i]

    def eval(self, X):
        return self(X), np.power(X, np.arange(4)[::-1]), None

    def __call__(self, X):
        self.inc()
        y = self.weights[0]
        for w in self.weights[1:]:
            y = y * X + w
        return y


class LinearConstrained(Objective):
    def __init__(self, X, y, lam=1.0):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        super().__init__(X.shape[1])

        # modified quadratic term
        n = len(self)
        I = np.eye(n)
        A = np.block([X, np.zeros_like(X)])
        self.H = A.T @ A

        # modified linear term
        c = np.vstack((np.zeros((n,1)), np.ones((n,1))))
        self.b = lam * c - A.T @ y

        # modified bias
        self.c = 0.5 * (y.T @ y)[0,0]

        # linear inequalities
        self.B = np.block([[I, -I], [-I, -I]])

    def __call__(self, X):
        self.inc()
        return (X.T @ (0.5 * self.H @ X + self.b))[0,0] + self.c

    def eval(self, X):
        # gradient
        gradL = self.H @ X + self.b

        # return function value, gradient and Hessian
        return self(X), gradL, self.H

    def ineq(self, X):
        return self.B @ X, self.B


def makeFromFun(Base, X, y, *args, **kwargs):
    class Derived(Base):
        def __init__(self):
            super().__init__(*args, **kwargs)

            self.X = X
            self.y = y

        def eval(self):
            f, df, _ = super().eval(self.X)
            return self.y - f, df, None
    return Derived

def makeFromOde(Base, X, y, *args, **kwargs):
    class Derived(Base, Objective):
        def __init__(self, dt):
            super().__init__(*args, **kwargs)
            self.nEval = 0

            self.dt = dt
            self.T = X
            self.X = y

        def __call__(self, x):
            self.inc()
            X = self.run(x, self.dt, self.T)
            return np.sum((X - self.X)**2)
    return Derived