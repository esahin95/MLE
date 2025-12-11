# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:31:31 2025

@author: sahin
"""

import numpy as np 
import matplotlib.pyplot as plt 
from MLTools import LogisticRegression

def lec07():
    # ground truth
    a, b, r = 0.5, 0.8, 1.0
    def f(X):
        y = np.ones((X.shape[0], 1))
        y[(X[:,0]/a)**2 + (X[:,1]/b)**2 > r**2] = -1.0
        return y
    
    # generate data
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(40,2))
    y = f(X)
    
    # feature map
    X = np.hstack((X, X**2))
    
    # train model
    model = LogisticRegression(X.shape[-1])
    model.fit(X, y, alp=1.0, epochs=10)
    
    # prediction
    Xt,Yt = np.meshgrid(np.linspace(0,1,1000), np.linspace(0,1,1000))
    Pt = np.vstack((Xt.flat, Yt.flat)).T
    
    yPt = model(np.hstack((Pt, Pt**2)))
    
    # post processing
    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    y = model(X)
    P = y.flatten() > 0.0
    N = y.flatten() < 0.0
    ax.contourf(
        Xt, Yt, np.reshape(yPt, Xt.shape), 
        cmap='coolwarm', 
        alpha=0.5, 
        vmin=-1.0, vmax=1.0
    )
    ax.scatter(X[P,0], X[P,1], color='r')
    ax.scatter(X[N,0], X[N,1], color='b')
    ax.set_axis_off()
    plt.savefig('Results/ellipse.pdf', bbox_inches='tight', pad_inches=0.0, transparent=True)
    
    
if __name__ == "__main__":
    lec07()