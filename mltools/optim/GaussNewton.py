import numpy as np
from . import timeit
from .Optimizer import Optimizer

class GaussNewton(Optimizer):
    def __step(self, f):
        b, A, _ = f.eval()
        f.add(np.linalg.lstsq(A, b)[0])
        return 0.5 * np.sum(b**2)

    @timeit
    def fit(self, f, x0=None, *, maxIter=10, eps=1e-5):
        if x0 is not None:
            f.set(x0)
        for i in range(maxIter):
            residual = self.__step(f)
            if residual < eps:
                break
        print(f'terminated at iteration {i}')