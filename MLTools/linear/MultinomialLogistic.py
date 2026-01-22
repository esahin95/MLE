import numpy as np
from .Base import Base
from . import timeit

class MultinomialLogistic(Base):
    def softmax(self, x):
        h = np.exp(x)
        return h / np.sum(h, axis=-1, keepdims=True)
    
    @timeit
    def fit(self, X, y, alp=0.01, epochs=1):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # one shot encoding
        eye = np.eye(self.weights.shape[-1])
        yEncoded = eye[y.flat]
        yIndex = (range(y.size), y.flat)
        
        # batch gradient descent
        for epoch in range(epochs):
            # residuals
            H = self.softmax(X @ self.weights)
            
            # gradient
            gradL = X.T @ (H - yEncoded)
            
            # direction
            d = -gradL / np.linalg.norm(gradL)
            
            # update
            #self.weights -= alp * gradL
            self.weights += alp * d
            
            # training progress
            if epoch % (epochs // 5) == 0:
                print(-np.sum(np.log(H[*yIndex])))
            
    def confusion(self, X, yTrue):
        C = np.zeros((self.weights.shape[-1], self.weights.shape[-1]))
        yPred = self(X)
        for i in range(len(X)):
            C[yPred[i,0], yTrue[i,0]] += 1
        return C
            
    def __call__(self, X):
        return np.argmax(self.softmax(super().__call__(X)), axis=-1, keepdims=True)