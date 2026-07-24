import numpy as np
from . import timeit
from .Optimizer import Optimizer


class NealderMead(Optimizer):
    def __init__(self, alp=0.5, bet=2.0, gam=1.0):
        super().__init__()
        self._alp = alp
        self._bet = bet
        self._gam = gam

    def __step(self, f):
        # find minimum and maximum
        n = len(f)
        m = np.argmax(self.__F)
        l = np.argmin(self.__F)

        # sorted array for convenience
        Fs = sorted(self.__F)

        # reflect on centroid
        sm = (np.sum(self.__S, axis=0) - self.__S[m]) / n
        xr = sm + self._gam * (sm - self.__S[m])
        fr = f.eval(xr)

        # case distinction
        if fr < Fs[0]:
            xe = sm + self._bet * (xr - sm)
            fe = f.eval(xe)
            if fe < fr:
                self.__S[m] = xe
                self.__F[m] = fe
            else:
                self.__S[m] = xr
                self.__F[m] = fr

        elif fr <= Fs[-2]:
            self.__S[m] = xr
            self.__F[m] = fr

        else:
            if fr >= Fs[-1]:
                xc = sm + self._alp * (self.__S[m] - sm)
            else:
                xc = sm + self._alp * (xr - sm)

            fc = f.eval(xc)
            if fc < Fs[-1]:
                self.__S[m] = xc
                self.__F[m] = fc
            else:
                for j in range(n+1):
                    if j == l:
                        continue
                    else:
                        self.__S[j] = 0.5 * (self.__S[j] + self.__S[l])

        return np.std(self.__F)


    @timeit
    def fit(self, f, x0=None, *, maxIter=10, eps=1e-3):
        if x0 is not None:
            f.set(x0)
        n = len(f)

        self.__S = np.eye(n+1,n) + f.state()
        self.__F = [f.eval(s) for s in self.__S]

        for i in range(maxIter):
            residual = self.__step(f)
            if residual < eps:
                break
        print(f'i = {i},  r = {residual:.5e}, n = {f.nEval()}')