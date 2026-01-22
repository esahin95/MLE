import numpy as np
from .Base import Base

class Logistic(Base):          
    def logit(self, x):
        return 1. / (1. + np.exp(-x))
    
    def fit(self, X, y, alp, epochs):
        # append bias 
        X = self.append(X)
        
        # optimize objective
        for epoch in range(epochs):
            # prediction
            h = self.logit(y * (X @ self.weights)) 
            
            # gradient
            gradL = - X.T @ (y * (1 - h))
            
            # hessian
            HL = X.T @ ((1 - h) * h * X)
            
            # step direction
            d = np.linalg.solve(HL, -gradL)
            
            # update
            self.weights += alp * d
            
            # training progress
            if epoch % (epochs // 5) == 0:
                print(-np.sum(np.log(h)))
        
    def __call__(self, X):
        # nonlinear transformation of linear prediction
        y = self.logit(super().__call__(X))
        
        # return most probable result
        return np.round(y) * 2 - 1