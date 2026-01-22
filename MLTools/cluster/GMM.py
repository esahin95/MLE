import numpy as np
from . import timeit

from scipy.stats import multivariate_normal as mn

class GMM:
    def __init__(self, X, k):
        self._X = X
        self._k = k
                
        # initialize parameters
        rng = np.random.default_rng(1000)
        m, d = X.shape
        self._phi = np.ones(k) / k
        self._mus = rng.choice(X, size=k, replace=False)
        self._sigmas = np.tile(np.eye(d), (k,1,1))
        self._W = np.zeros((m, k))
        
    def fit(self, maxIter=20):
        m, d = self._X.shape

        for iteration in range(maxIter):
            # E step
            p = [mn(mean=self._mus[j], cov=self._sigmas[j]) for j in range(k)]
            for j in range(k):
                p = mn(mean=self._mus[j], cov=self._sigmas[j])
                self._W[:,j] = p.pdf(self._X)
            self._W = self._W * self._phi
            print(np.sum(np.log(np.sum(self._W, axis=1))))
            self._W /= np.sum(self._W, axis=1, keepdims=True)
            
            # M step
            sumW = np.sum(self._W, axis=0)
            self._phi = sumW / m
            for j in range(k):
                self._mus[j] = (self._W[:,j] @ self._X) / sumW[j]
                self._sigmas[j] = (self._X - self._mus[j]).T @ (self._W[:,j:j+1] * (self._X - self._mus[j])) / sumW[j]
            
    def predict(self, X):
        m, d = X.shape
        
        W = np.zeros((m, self._k))
        for j in range(self._k):
            p = mn(mean=self._mus[j], cov=self._sigmas[j])
            W[:,j] = p.pdf(X)
        return np.sum(W * self._phi, axis=1, keepdims=True)