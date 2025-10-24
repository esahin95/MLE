# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:32:47 2025

@author: sahin
"""

import numpy as np

class NaiveBayes:
    def fit(self, X, Y):
        # number of cases
        m, n = X.shape
        nClass1 = np.sum(Y)
        nClass0 = m - nClass1
        
        # estimate model parameters
        self._Py = nClass1 / m
        self._PI = np.zeros((2, n))
        for i in range(n):
            self._PI[1,i] = (1 + np.sum(X[Y.flat == 1,i])) / (nClass1 + 2)
            self._PI[0,i] = (1 + np.sum(X[Y.flat == 0,i])) / (nClass0 + 2)
            
    def prob(self, X):
        P = np.ones((X.shape[0], 2)) * np.array([[1-self._Py, self._Py]])
        for i in range(2):
            P[:,i:i+1] *= np.prod(self._PI[i:i+1]**X * (1 - self._PI[i:i+1])**(1 - X), axis=-1, keepdims=True)
        return P
    
    def predict(self, X):
        return np.argmax(self.prob(X), axis=-1, keepdims=True).astype(np.int8)
    
    def confusion(self, X, Y):
        P = self.predict(X)
        K = np.zeros((2,2))
        for p, y in zip(P, Y):
            K[p,y] += 1
        return K
    
class NaiveBayesMixed(NaiveBayes):
    def fit(self, X, Y):
        Xnom, Xcar = X
        
        # probabilities for nominal scales
        super().fit(Xnom, Y)
        
        # gauss distribution for cardinal scales
        self._mu = np.zeros((2, Xcar.shape[-1]))
        self._std = np.zeros_like(self._mu)
        for i in range(2):
            self._mu[i] = np.mean(Xcar[Y==i])
            self._std[i] = np.std(Xcar[Y==i])
            
    def prob(self, X):
        Xnom, Xcar = X 
        
        P = super().prob(Xnom)
        P *= np.exp(-0.5*(Xcar - self._mu.T)**2/self._std.T**2) / (self._std.T*np.sqrt(2*np.pi))
        return P
    
class ICA:
    def __getitem__(self, idx):
        return self._W[idx]
    
    def fit(self, X, bs=100, lr=5e-4, epochs=10):
        n, m = X.shape
        rng = np.random.default_rng()
        
        self._W = np.eye(n)
        for epoch in range(epochs):
            I = rng.permutation(range(m))
            for j in range(m//bs):
                Xbatch = X[:,I[j*bs:j*bs+bs]]
                Gbatch = 1.0 - 2.0 / (1.0 + np.exp(- self._W @ Xbatch))
                self._W += lr * (Gbatch @ Xbatch.T + bs * np.linalg.inv(self._W.T))
                
        return self._W @ X