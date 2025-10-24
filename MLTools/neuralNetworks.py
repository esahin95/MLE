# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:26:18 2024

@author: emres
"""

import numpy as np
    
class Layer:
    def __init__(self, nInputs, nOutputs, rng):
        self._W = rng.standard_normal((nOutputs, nInputs))
        self._b = rng.standard_normal(nOutputs)
        
    def __call__(self, X):
        self._X = X
        H = X @ self._W.T + self._b
        H, self._dH = self.activation(H)
        return H
    
    def __repr__(self):
        s = f'{self.__class__!s} with:\n'
        s += f'weights of shape {self._W.shape!r} \n {self._W!r} \n'
        s += f'bias of shape {self._b.shape!r} \n {self._b!r} \n'
        return s
    
    def backprop(self, dY, alp):
        # derivatives of hidden variables
        self._dH *= dY
        
        # derivatives of inputs
        self._Xgrad = self._dH @ self._W
        
        # average derivative of bias
        self._bgrad = np.sum(self._dH, axis=0)
        self._b -= alp * self._bgrad
        
        # average derivative of weights
        self._Wgrad = self._dH.T @ self._X
        self._W -= alp * self._Wgrad
        
        # propagate derivatives
        return self._Xgrad

class Linear(Layer):
    def activation(self, X):
        return X, np.ones_like(X)

class Sigmoid(Layer):
    def activation(self, X):
        Z = 1 / (1 + np.exp(-X))
        return Z, Z * (1 - Z)
    
class MLP:
    def __init__(self, architecture, seed=None):
        self._nFeatures = architecture[0]
        self._nTargets  = architecture[-1]      
        rng = np.random.default_rng(seed)
        
        # define layers
        self._layers = []
        for i in range(len(architecture) - 2):
            self._layers.append(Sigmoid(architecture[i], architecture[i+1], rng))
        self._layers.append(Linear(architecture[-2], architecture[-1], rng))
        
    def __call__(self, X):
        for layer in self._layers:
            X = layer(X)
        return X
    
    def __repr__(self):
        return f'{self.__class__!s} has layers:\n{self._layers!r}'
    
    def backprop(self, dY, alp):
        for layer in self._layers[::-1]:
            dY = layer.backprop(dY, alp)
        return dY
            
    def fit(self, X, Y, lr=0.01, bs=5, epochs=20):
        m = X.shape[0]
        n = m // bs 
        L = []
        for i in range(epochs):
            L.append(0)
            for j in range(n):
                # get batch
                Xb, Yb = X[bs*j:bs*(j+1), :], Y[bs*j:bs*(j+1), :]
            
                # residuals
                R = (self(Xb) - Yb)
            
                # loss function
                L[-1] += np.sum(R**2)
                
                # error backpropagation
                self.backprop(R*2.0/Yb.size, lr)
            
            L[-1] /= Yb.size
            if i % max(1, epochs//5) == 0:
                print(f'current loss {L[-1]:.5e}')
        return L