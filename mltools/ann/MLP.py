import numpy as np
from .Sigmoid import Sigmoid
from .Linear import Linear

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