import numpy as np

class Base:
    def __init__(self, nFeatures=1, nTargets=1, bias=True):
        self.bias = bias
        self.weights = np.zeros((nFeatures + bias, nTargets))
        
    def __call__(self, X):
        if self.bias:
            y = X @ self.weights[1:] + self.weights[0]
        else:
            y = X @ self.weights
        return y
    
    def __repr__(self):
        return f'{self.weights!r}'
    
    def append(self, X):
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X