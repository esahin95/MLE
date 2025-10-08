# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:33:25 2025

@author: sahin
"""

import numpy as np
import matplotlib.pyplot as plt
from MLTools import *

def run01():
    ''' 
    Run code for exercise 01. Predict falling time of a point mass with 
    linear regression model. Data generated with ODE integrator
    '''
    # seeded random number generator
    rng = np.random.default_rng(0)
    n = 20
    
    # random initial conditions
    v = rng.uniform(-0.5, 0.5, (n,1))
    h = rng.uniform( 0.1, 2.0, (n,1))
    
    # environment
    g = 9.81
    
    # integrate ODE
    ode = FreeFall(g)
    t = np.zeros((n,1))
    for i in range(n):
        t[i] = ode.run(np.array([h[i], v[i]]), 1e-3)
    
    # nondimensionalized input
    X = v / np.sqrt(g * h)
    
    # nondimensionalized output
    y = t * np.sqrt(g / h)
    
    # build model
    model = LinearRegression(nFeatures=1)
    model.fit(X, y)
    print(model)
    
    # test on moon
    g, h, v = 1.62, 1.5, 0.1
    x = v / np.sqrt(h * g)
    tTrue = np.sqrt(h / g) * (x + np.sqrt(x**2 + 2.0))
    tPred = np.sqrt(h / g) * model(x.reshape(1,1))
    print(tTrue, tPred)
    
    
def run02():
    ''' 
    Run code for exercise 02. Predict part failure for given temperature. 
    Test Perceptron and logistic regression model. Perceptron fails, since 
    data is not linearly seperable
    '''
    test = "Logistic"
    
    # dataset from MLE book
    data = DataCollection()
    data.load("Data/partFailure.npz")
    print(data.X, data.y, sep='\n')
    
    # build model
    match test:
        case "Logistic":
            model = LogisticRegression(1)
            model.fit(data.X, data.y, alp=1.0, epochs=10)
            
        case "Perceptron":
            model = Perceptron(1)
            model.fit(data.X, data.y, alp=0.1, epochs=300)
            
        case _:
            raise ValueError("unknown model type")
    print(model)
    
    # compare prediction
    plt.scatter(data.X, data.y, c='b')
    plt.scatter(data.X, model(data.X), facecolors='none', edgecolors='r')
    plt.show()
    

def run03():
    '''
    Additional content for exercise 02. Multi-class classification problem
    with multinomial logistic regression
    '''
    # generate synthetic dataset
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10,1))
    y = ((X>0.5) * 2 + (X<-0.5) * 1).astype(np.int32)
    
    # build model
    model = MultinomialLogisticRegression(1, 3)
    model.fit(X, y, alp=1.0, epochs=100)
    
    # post processing
    plt.scatter(X,y)
    plt.scatter(X, model(X), facecolors='none', edgecolors='r')
    plt.show()


if __name__ == "__main__":
    #run01()
    run02()
    #run03()