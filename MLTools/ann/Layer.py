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