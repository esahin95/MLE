# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:25:55 2024

@author: emres
"""

import numpy as np

class Linear:
    def __init__(self, nFeatures=1):
        ''' 
        Weights of linear objective with additional variable for bias
        '''
        self._w = np.zeros(nFeatures + 1)
    
    def fit(self, X, y, alp=1.0, maxit=20):
        ''' 
        fit weights to minimize loss function. Step function in derived classes
        '''
        # append bias
        X = np.hstack((np.ones((y.size, 1)), X))
        
        # optimize objective
        for i in range(maxit):
            self.update(X, y, alp)
        
    def state(self):
        ''' 
        Access weights
        '''
        return self._w 
    
    def __call__(self, X):
        ''' 
        Linear prediction with no output transformation
        '''
        # linear prediction
        return X @ self._w[1:] + self._w[0]
    
class Perceptron(Linear):
    def update(self, X, y, alp):
        ''' 
        incremental updates
        '''
        for xi, yi in zip(X, y):
            yxi = yi * xi
            if yxi @ self._w <= 0:
                self._w += alp * yxi
                
    def __call__(self, X):
        # return sign of linear prediction as label
        return np.sign(super().__call__(X))

class LinearRegression(Linear):        
    def update(self, X, y, alp):
        ''' 
        Implements newton method for linear model with squared residual loss
        '''
        # linear prediction
        z = X @ self._w
        
        # loss function
        L = 0.5 * np.sum((z - y)**2)
        print(f'current loss: {L:.4e}')
        
        # gradient
        gradL = X.T @ (z - y) 
        
        # hessian
        HL = X.T @ X
        
        # step direction
        d = np.linalg.solve(HL, -gradL) 
        
        # update weights
        self._w += alp * d
    
class LogisticRegression(Linear):          
    def logit(self, x):
        ''' 
        Sigmoid function for arbitrary input shapes
        '''
        return 1. / (1. + np.exp(-x))
      
    def update(self, X, y, alp):
        ''' 
        Implements newton method for logistic regression with negative log likelihood
        '''
        # prediction
        h = self.logit(y * (X @ self._w)) 
        
        # loss function
        L = - np.sum(np.log(h))
        print(f'current loss: {L:.4e}')
        
        # gradient
        gradL = - X.T @ (y * (1 - h))
        
        # hessian
        HL = (X.T * (1-h) * h) @ X

        # step direction
        d = np.linalg.solve(HL, -gradL) 
        
        # update weights
        self._w += alp * d
        
    def __call__(self, X):
        ''' 
        Linear prediction with nonlinear transformation to predicted labels
        '''
        # nonlinear transformation of linear prediction
        y = self.logit(super().__call__(X))
        
        # return most probable result
        return np.round(y)*2 - 1