import numpy as np
from .Layer import Layer

class Sigmoid(Layer):
    def activation(self, X):
        Z = 1 / (1 + np.exp(-X))
        return Z, Z * (1 - Z)