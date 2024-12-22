# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:26:18 2024

@author: emres
"""

import numpy as np

class Layer:
    def __init__(self, nInputs, nOutputs, rng=None):
        ''' 
        Random initialization of training parameters
        '''
        if rng is None:
            rng = np.random.default_rng()
        self._W = rng.standard_normal((nInputs, nOutputs))
        self._b = rng.standard_normal(nOutputs)
    
    def update(self, dY, alp):
        ''' 
        Error backpropagation through layer and GD update step
        '''
        # batch size
        nSamples = self._X.shape[0]
        
        # derivatives of hidden variables
        self._dH *= dY
        
        # derivatives of inputs
        dX = self._dH @ self._W.T
        
        # average derivatie of bias
        self._b -= alp * np.sum(self._dH, axis=0) / nSamples
        
        # average derivative of weights
        self._W -= alp * self._X.T @ self._dH / nSamples
        
        # propagate derivatives
        return dX
    
    def __repr__(self):
        ''' 
        Representation of MLP class
        '''
        s = f'{self.__class__!s} with:\n'
        s += f'weights of shape {self._W.shape!r} \n {self._W!r} \n'
        s += f'bias of shape {self._b.shape!r} \n {self._b!r} \n'
        return s
    
    def __call__(self, X):
        ''' 
        Forward propagation of inputs
        '''
        self._X = X
        H = X @ self._W + self._b
        H, self._dH = self.activation(H)
        return H
    
class Linear(Layer):
    def activation(self, X):
        ''' 
        Linear layer, mainly for output layer
        '''
        return X, np.ones_like(X)
    
class Sigmoid(Layer):
    def activation(self, X):
        ''' 
        Nonlinear layer, mainly for hidden layers
        '''
        Z = 1 / (1 + np.exp(-X))
        return Z, Z * (1 - Z)
    
class MLP:
    def __init__(self, arch, seed=None):
        ''' 
        Build feed forward network given by architecture (arch)
        '''
        self.arch = arch.copy()
        self._rng = np.random.default_rng(seed)
        self._layers = []
        for i in range(len(arch) - 2):
            self._layers.append(Sigmoid(arch[i], arch[i+1], self._rng))
        self._layers.append(Linear(arch[-2], arch[-1], self._rng))
    
    def update(self, dY, alp):
        ''' 
        Backpropagation
        '''
        for layer in self._layers[::-1]:
            dY = layer.update(dY, alp)
    
    def __call__(self, X):
        ''' 
        Forward propagation
        '''
        for layer in self._layers:
            X = layer(X)
        return X
    
    def __repr__(self):
        ''' 
        Representation of MLP class
        '''
        return f'{self.__class__!s} has layers:\n{self.layers!r}'
    
    def fit(self, X, Y, alp=0.01, epochs=20):
        ''' 
        Training procedure for MLP
        '''
        nSamples = X.shape[0]
        batchSize = 5
        nBatches = X.shape[0] // batchSize
        L = []
        for i in range(epochs):
            L.append(0)
            for j in range(nBatches):
                Xb, Yb = X[batchSize*j:batchSize*(j+1), :], Y[batchSize*j:batchSize*(j+1), :]
            
                # current residuals
                R = self(Xb) - Yb 
            
                # loss function
                L[-1] += 0.5 * np.sum(R**2)
                
                # error backpropagation
                self.update(R, alp)
            
            L[-1] /= nSamples
        return L