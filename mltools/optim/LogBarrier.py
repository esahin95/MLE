import numpy as np
from . import timeit
from .Optimizer import Optimizer

class LogBarrier(Optimizer):
    def __init__(self, f):
        # reference to objective
        self._f = f

        # initial solution
        d = len(f)
        self.x = np.vstack((np.zeros((d,1)), np.ones((d,1))))


    def eval(self, x):
        # inequalities
        Bx, B = self._f.ineq(x)

        # objective
        _, gradF, HF = self._f.eval(x)

        # add log barrier
        gradL = gradF - self._mu * B.T @ (Bx**-1)
        HL = HF + self._mu * B.T @ ((Bx**-2) * B)

        # return gradient and Hessian
        return gradL, HL

    def __call__(self, x):
        # check inequalities
        q = self._f.ineq(x)[0]
        if np.any(q >= 0):
            return np.inf

        # add log barrier
        return self._f(x) - self._mu * np.sum(np.log(-q))

    def step(self, x, maxIter):
        alp = 1.0
        for i in range(maxIter):
            # Newton direction
            gradL, HL = self.eval(x)
            d = np.linalg.solve(HL, -gradL)

            # line search
            k0, k1 = self(x), 0.01 * np.sum(gradL * d)
            while self(x + alp*d) > k0 + k1 * alp:
                alp *= 0.5
                if alp < 1e-5:
                    raise Exception('alpha too small')
            d *= alp
            alp = min(1.0, 1.2*alp)
            x += d

            # termination condition
            if np.linalg.norm(d, np.inf) < 1e-5:
                return x
        raise Exception('Newton did not converge')

    @timeit
    def fit(self, f, x0=None, *, epochs=50, maxIter=50):
        if x0 is None:
            self.set(x0)

        self._mu = 1.0
        for epoch in range(epochs):
            x = self.x.copy()
            x = self.step(x, maxIter)

            # termination condition
            if np.linalg.norm(x - self.x, np.inf) < 1e-4:
                self.x = x
                print(f'terminated at iteration {epoch} and residual {self._f(x)}')
                break

            # setup for next iteration
            self._mu *= 0.5
            self.x = x