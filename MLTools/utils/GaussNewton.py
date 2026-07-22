import numpy as np
from . import timeit

class GaussNewton:
    def step(self, f):
        # objective value and gradient
        b, A = f.eval()
        r = 0.5 * np.sum(b**2)

        # regularized least squares problem
        d = np.linalg.lstsq(A,b)[0]

        # update parameters
        f.weights += d

        # return residual
        return r

    @timeit
    def fit(self, f, maxIter=10, eps=1e-5):
        for i in range(maxIter):
            # solve subproblem
            if self.step(f) < eps:
                break
        print(f'terminated at iteration {i}')