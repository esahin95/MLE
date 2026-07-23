import numpy as np
from . import timeit

class GaussNewton:
    def step(self, f):
        b, A, _ = f.eval()

        f.weights += np.linalg.lstsq(A, b)[0].flatten()

        return 0.5 * np.sum(b**2)

    @timeit
    def fit(self, f, maxIter=10, eps=1e-5):
        for i in range(maxIter):
            if self.step(f) < eps:
                break
        print(f'terminated at iteration {i}')