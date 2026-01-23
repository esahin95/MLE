# imports
import sys
sys.path.append("..\\..\\")

import numpy as np
import matplotlib.pyplot as plt

from mltools.svm import SVC, SVCSE

# ground truth
a, b, c = 2.0, 0.0, 0.5
def f(x):
    return c * np.sin(a * x - b)
def F(X):
    y = np.ones((X.shape[0], 1))
    y[X[:,1] < f(X[:,0])] = -1
    return y

# generate random data
rng = np.random.default_rng(10)
X = rng.uniform(-1,1,size=(120,2))
y = F(X)

# train model
model = SVCSE(sigma=0.5)
model.fit(X, y, C=50, tol=1e-4, maxPasses=5)

# prediction
Xt,Yt = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
Pt = np.vstack((Xt.flat, Yt.flat)).T
yPred = model.predict(Pt)

# postprocessing
fig, ax = plt.subplots(1, 1, figsize=(3,3))
P = y.flatten() > 0.0
N = y.flatten() < 0.0
ax.contourf(
    Xt, Yt, np.reshape(yPred, Xt.shape), 
    cmap='coolwarm', 
    alpha=0.5, 
    vmin=-1.0, vmax=1.0
)
ax.scatter(X[P,0], X[P,1], s=5, color='r')
ax.scatter(X[N,0], X[N,1], s=5, color='b')
ax.set_axis_off()
x = np.linspace(-1,1,100)
plt.plot(x, f(x), 'k')
plt.show()