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
        self.weights, residuals = np.linalg.lstsq(X, y)[0:1]
        print(residuals)
    
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
                
class RidgeRegression(Linear):
    def fit(self, X, y, lam=0.01):
        m, d = X.shape
        
        # append bias 
        X = np.hstack((np.ones((m, 1)), X))
        
        # append penalty
        X = np.vstack((X, np.sqrt(lam)*np.eye(d+1)))
        y = np.vstack((y, np.zeros((d+1, 1))))
        
        # solve least squares
        self.weights, residuals, rank, s = np.linalg.lstsq(X, y)
        print(residuals.item())
        
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
        
class WeightedLogistic(LogisticRegression):    
    def fit(self, X, y, tau, lam=1e-4):
        self._X = np.hstack((np.ones((X.shape[0], 1)), X)) 
        self._y = y.copy() 
        self._scl = 2.0 * tau**2
        self._lam = lam
        self._lamI = lam * np.eye(X.shape[-1] + 1)
        
    def predict(self, x, maxIter=50):
        for i in range(maxIter):
            # distances
            d = np.exp(-np.sum((self._X - x)**2, axis=-1, keepdims=True) / self._scl)
            
            # logistics
            g = self.logit(self._y * (self._X @ self.weights))
            
            # gradient
            gradL = self._lam * self.weights - self._X.T @ (d * (1 - g) * self._y)
            
            # Hessian
            HL = self._lamI + self._X.T @ ((d * g * (1-g)) * self._X)
            
            # direction
            d = np.linalg.solve(HL, -gradL)
            
            # update
            self.weights += d 
            
            # termination condition
            if np.linalg.norm(gradL) < 1e-5:
                return super().__call__(x[:,1:])[0,0]
        raise Exception('Newton did not converge')
        
    def __call__(self, X):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # predict for each x
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            self.weights *= 0.0
            y[i] = self.predict(X[i:i+1])
        return y