# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:25:55 2024

@author: emres
"""

import numpy as np
from . import timeit

class Linear:
    def __init__(self, nFeatures=1, nTargets=1):
        self.weights = np.zeros((nFeatures + 1, nTargets))
        
    def __call__(self, X):
        return X @ self.weights[1:] + self.weights[0]
    
    def __repr__(self):
        return f'{self.weights!r}'

    
class Perceptron(Linear):
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

class LinearRegression(Linear):
    def fit(self, X, y):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # optimize objective
        self.weights = np.linalg.lstsq(X, y)[0]
    
class LogisticRegression(Linear):          
    def logit(self, x):
        return 1. / (1. + np.exp(-x))
    
    def fit(self, X, y, alp, epochs):
        # append bias 
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
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
    
class MultinomialLogisticRegression(Linear):
    def softmax(self, x):
        h = np.exp(x)
        return h / np.sum(h, axis=-1, keepdims=True)
    
    def fit(self, X, y, alp=0.01, epochs=1):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # one shot encoding
        eye = np.eye(self.weights.shape[-1])
        yEncoded = eye[y.flat]
        
        # batch gradient descent
        for epoch in range(epochs):
            # residuals
            h = self.softmax(X @ self.weights)
            
            # gradient
            gradL = X.T @ (h - yEncoded)
            
            # update
            self.weights -= alp * gradL
            
            # training progress
            if epoch % (epochs // 5) == 0:
                # loss
                print(-np.sum(np.log(h[i][y[i,0]]) for i in range(len(h))))
                
    def __call__(self, X):
        return np.argmax(self.softmax(super().__call__(X)), axis=-1, keepdims=True)
                
class RidgeRegression(Linear):
    def fit(self, X, y, lam=0.01):
        m, d = X.shape
        
        # append bias 
        X = np.hstack((np.ones((m, 1)), X))
        
        # append penalty
        X = np.vstack((X, np.sqrt(lam)*np.eye(d+1)))
        y = np.vstack((y, np.zeros((d+1, 1))))
        
        # solve least squares
        self.weights = np.linalg.lstsq(X, y)[0]
        
class Lasso(Linear):
    @timeit
    def fit(self, X, y, lam=0.01, epochs=50):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # optimization loop
        for epoch in range(epochs):
            dw = 0
            # coordinate ascent
            for i in range(X.shape[-1]):                
                # eliminate coordinate
                w, self.weights[i,0] = self.weights[i,0], 0
                
                # reduced costs 
                J = y - X @ self.weights
                a = np.sum(X[:,i]**2)
                b = (X[:,i] @ J)[0]
                
                # potential solutions
                wip = max(0, (b - lam) / a)
                win = min(0, (b + lam) / a)
                
                # optimal solution
                Jp = 0.5*a*wip**2 - (b - lam)*wip
                Jn = 0.5*a*win**2 - (b + lam)*win
                if Jp < Jn:
                    self.weights[i] = wip
                else:
                    self.weights[i] = win
                    
                # maximum change
                dw = max(dw, np.abs(w - self.weights[i,0]))
                
            # current loss
            if dw < 1e-5:
                L = 0.5 * np.sum((X @ self.weights - y)**2) + lam * np.linalg.norm(self.weights, 1)
                print(f'terminated at iteration {epoch} with residual {L}')
                return
        raise Exception('Lasso failed to converge')