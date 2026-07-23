import numpy as np
from . import timeit

class NealderMead:
    def __init__(self, alp=0.5, bet=2.0, gam=1.0):
        self._alp = alp
        self._bet = bet
        self._gam = gam

    def __call__(self, fun, x, eps=1e-3):
        # initial simplex
        n = x.size
        S = np.eye(n+1,n) + x
        F = [fun(s) for s in S]

        for i in range(1000):
            # find minimum and maximum
            m = np.argmax(F)
            l = np.argmin(F)

            # sorted array for convenience
            Fs = sorted(F)

            # reflect on centroid
            sm = (np.sum(S, axis=0) - S[m]) / n
            xr = sm + self._gam * (sm - S[m])
            fr = fun(xr)

            # case distinction
            if fr < Fs[0]:
                xe = sm + self._bet * (xr - sm)
                fe = fun(xe)
                if fe < fr:
                    S[m] = xe
                    F[m] = fe
                else:
                    S[m] = xr
                    F[m] = fr

            elif fr <= Fs[-2]:
                S[m] = xr
                F[m] = fr

            else:
                if fr >= Fs[-1]:
                    xc = sm + self._alp * (S[m] - sm)
                else:
                    xc = sm + self._alp * (xr - sm)

                fc = fun(xc)
                if fc < Fs[-1]:
                    S[m] = xc
                    F[m] = fc
                else:
                    for j in range(n+1):
                        if j == l:
                            continue
                        else:
                            S[j] = 0.5 * (S[j] + S[l])

            # termination criteria
            if np.std(F) < eps:
                print(S[l], Fs[-1], fun.nEval)
                return S[l]