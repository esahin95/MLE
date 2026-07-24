import numpy as np
from .Base import Base

class Perceptron(Base):
    def fit(self, X, y, alp, epochs):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # optimize objective
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                if (yi * xi) @ self.weights <= 0:
                    self.weights.flat += (alp * yi) * xi
        self.weights /= np.linalg.norm(self.weights, np.inf) 
                
    def __call__(self, X):
        # return sign of linear transformation as label
        return np.sign(super().__call__(X))