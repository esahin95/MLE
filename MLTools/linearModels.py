# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:25:55 2024

@author: emres
"""

import numpy as np

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
            miss = 0
            for xi, yi in zip(X, y):
                # perceptron learning rule
                yxi = yi * xi
                if yxi @ self.weights <= 0:
                    self.weights.flat += alp * yxi
                    miss += 1
            print(miss / X.shape[0], end=' ')
                    
        # normalization
        self.weights /= np.linalg.norm(self.weights, np.inf) 
        print()
                
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
                # loss
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
        # append bias 
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # append penalty
        L = np.sqrt(lam) * np.eye(X.shape[-1])
        X = np.vstack((X, L))
        y = np.vstack((y, np.zeros((X.shape[-1], 1))))
        
        # solve least squares
        self.weights = np.linalg.lstsq(X, y)[0]