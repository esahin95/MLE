import numpy as np
from .Layer import Layer

class Linear(Layer):
    def activation(self, X):
        return X, np.ones_like(X)