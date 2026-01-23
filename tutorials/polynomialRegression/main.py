# imports
import sys
sys.path.append("..\\..\\")

import numpy as np
import matplotlib.pyplot as plt

from mltools.utils import LogBarrier
from mltools.linear import Ridge

# ground truth
def f(X):
    return 3*X - X**2 + 2*X**3

# generate data
rng = np.random.default_rng(0)
m = 10
X = rng.uniform(-1, 1, size=(m,1))
y = f(X) + rng.normal(loc=0.0, scale=0.1, size=(m,1))

# feature map
p = 8
Phi = np.power(X[...,np.newaxis], np.arange(1,p+1)).reshape(X.shape[0],-1)

# objective
class Objective:
    def __init__(self, X, y, lam=1.0):
        # append bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # original problem size
        self._nDim = X.shape[-1]
        
        # modified quadratic term
        I = np.eye(self._nDim)
        A = np.block([X, np.zeros_like(X)])
        self._H = A.T @ A
        
        # modified linear term
        c = np.vstack((np.zeros((self._nDim,1)), np.ones((self._nDim,1))))
        self._b = lam * c - A.T @ y
        
        # modified bias
        self._c = 0.5 * (y.T @ y)[0,0]
        
        # linear inequalities
        self._B = np.block([[I, -I], [-I, -I]])
        
    def __call__(self, x):
        return (x.T @ (0.5 * self._H @ x + self._b))[0,0] + self._c
    
    def eval(self, x):
        # gradient          
        gradL = self._H @ x + self._b
        
        # return gradient and Hessian
        return gradL, self._H
        
    def ineq(self, x):
        return self._B @ x, self._B
    
    def __len__(self):
        return self._nDim
f = Objective(Phi, y, lam=0.1)

# constrained optimization    
logB = LogBarrier(f)
logB.optimize(epochs=50, maxIter=50)

# unconstrained optimization
linR = Ridge(nFeatures=p)
linR.fit(Phi, y, lam=0.1)

# post processing
x = np.arange(p + 1).reshape(-1,1)
fig, ax = plt.subplots(1,1,figsize=(4,2))
ax.scatter(x, logB.x[:p+1], s=15, color='r', marker='o')
ax.scatter(x, linR.weights, s=15, color='b', marker='o')
plt.show()