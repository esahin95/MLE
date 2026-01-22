# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:50:56 2025

@author: sahin
"""

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
                print(S[l], Fs[-1], fun._nEval)
                return S[l]
            
class GN:        
    def step(self, f):
        # objective value and gradient
        b, A = f.eval()
        r = 0.5 * np.sum(b**2)
        print(r)
        
        # regularized least squares problem
        d = np.linalg.lstsq(A,b)[0]
        
        # update parameters
        f.weights += d
        
        # return residual
        return r
        
    def fit(self, f, maxIter=10, eps=1e-5):
        for i in range(maxIter):
            # solve subproblem
            if self.step(f) < eps:
                break
        print(f'terminated at iteration {i}')
            
        
class LogBarrier:
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
        gradF, HF = self._f.eval(x)
        
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
            
            # termination condition
            x += d
            if np.linalg.norm(d, np.inf) < 1e-5:
                #print(f'terminated at iteration {i} with alpha {alp} and residual {self(x)}')
                return x
        raise Exception('Newton did not converge')
    
    @timeit
    def optimize(self, epochs=50, maxIter=50):
        self._mu = 1.0 
        for epoch in range(epochs):
            # initialize solution
            x = self.x.copy()
            
            # Newton optimization with line search
            x = self.step(x, maxIter)
            
            # termination condition
            if np.linalg.norm(x - self.x, np.inf) < 1e-4:
                self.x = x 
                print(f'terminated at iteration {epoch} and residual {self._f(x)}')
                break 
            
            # setup for next iteration
            self._mu *= 0.5
            self.x = x