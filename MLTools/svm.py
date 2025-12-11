# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 15:21:46 2025

@author: sahin
"""

import numpy as np 
from . import timeit

class SVC:
    @timeit      
    def fit(self, X, y, maxPasses=3, C=0.1, tol=1e-3):
        # problem size
        m = X.shape[0]
        
        # initialize support vector parameters
        self.alpha = np.zeros((m,1))
        self.bias = 0.0
        
        # build covariance matrix
        K = np.zeros((m,m))
        for i in range(m):
            for j in range(i,m):
                K[i,j] = self.K(X[i],X[j])
                K[j,i] = K[i,j]
        
        passes = 0
        while passes <= maxPasses:
            numAlphaChanged = 0
            
            for i in range(m):
                Ei = K[i] @ (self.alpha * y) + self.bias - y[i]
                if (y[i]*Ei < -tol and self.alpha[i] < C) or (y[i]*Ei > tol and self.alpha[i] > 0):
                    # select second index randomly
                    j = np.random.choice(m-1)
                    if j >= i:
                        j += 1
                    Ej = K[j] @ (self.alpha * y) + self.bias - y[j]
                    
                    # coordinate limits
                    if y[i] * y[j] < 0:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(C, C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[j] + self.alpha[i] - C)
                        H = min(C, self.alpha[j] + self.alpha[i])
                    if L == H:
                        continue
                    
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue
                    
                    # compute alphas
                    aj = self.alpha[j] - y[j] * (Ei - Ej) / eta 
                    if aj > H:
                        aj = H 
                    if aj < L:
                        aj = L
                    if np.abs(aj-self.alpha[j]) < 1e-5:
                        continue
                    ai = self.alpha[i] + y[i]*y[j]*(self.alpha[j] - aj)
                    
                    # update bias
                    b1 = self.bias - Ei - y[i]*(ai-self.alpha[i])*K[i,i] - y[j]*(aj-self.alpha[j])*K[i,j] 
                    b2 = self.bias - Ej - y[i]*(ai-self.alpha[i])*K[i,j] - y[j]*(aj-self.alpha[j])*K[j,j] 
                    if ai > L and ai < H:
                        self.bias = b1
                    elif aj > L and aj < H:
                        self.bias = b2
                    else:
                        self.bias = 0.5 * (b1 + b2)
                    
                    # update alphas
                    numAlphaChanged += 1
                    self.alpha[i] = ai
                    self.alpha[j] = aj
            
            # check if alpha was changed during run
            if numAlphaChanged == 0:
                passes += 1 
            else:
                passes = 0
            
        # save active set
        active = self.alpha.flat >= tol
        self.alpha = self.alpha[active]
        self.X = X[active]
        self.y = y[active]
                
    def K(self, x, z):
        return np.inner(x, z)
    
    def __call__(self, Z):
        n, m = Z.shape[0], self.X.shape[0]
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K[i,j] = self.K(Z[i], self.X[j])
        
        return K @ (self.alpha * self.y) + self.bias
    
    def predict(self, Z):
        return np.sign(self(Z))
                
class SVCSE(SVC):
    def __init__(self, sig=1.0):
        self._sig2 = sig**2
        
    def K(self, x, z):
        return np.exp(-0.5*np.linalg.norm(x-z)**2 / self._sig2)