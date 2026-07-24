import numpy as np
from .Objective import Objective


class LinearConstrained(Objective):
    def __init__(self, X, y, lam=1.0):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        super().__init__(n=X.shape[1])

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
        return (X.T @ (0.5 * self.H @ X + self.b))[0,0] + self.c

    def eval(self, X):
        gradL = self.H @ X + self.b

        return self(X), gradL, self.H

    def ineq(self, X):
        return self.B @ X, self.B