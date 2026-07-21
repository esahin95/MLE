# imports
import sys
sys.path.append("..\\..\\")

import numpy as np
import matplotlib.pyplot as plt

from mltools.ann import MLP

# ground truth
rng = np.random.default_rng(0)
base = MLP([2,2,2], seed=42)
X = rng.random((250,2))
Y = base(X)

# structured mesh
U,V = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
P = np.hstack((U.reshape(-1,1), V.reshape(-1,1)))
W = base(P)

# fit neural network
model = MLP([2,2,2], seed=659)
L = model.fit(X, Y, lr=0.01, bs=X.shape[0], epochs=100000)
YP = model(X)
WP = model(P)

# plot 
fig, axs = plt.subplots(1,2,figsize=(10,5), subplot_kw={'projection':'3d'})
for i in range(Y.shape[-1]):
    # training data
    axs[i].scatter(X[:,0:1], X[:,1:2], Y[:,i:i+1], c='b', s=1)
    axs[i].scatter(X[:,0:1], X[:,1:2], YP[:,i:i+1], c='r', s=1)
    
    # structured surface plot
    axs[i].plot_surface(U, V, W[:,i].reshape(U.shape), color='b', alpha=0.2)
    axs[i].plot_surface(U, V, WP[:,i].reshape(U.shape), color='r', alpha=0.2)
plt.show()

# training loss
fig, ax = plt.subplots(1,1)
ax.semilogy(L[0:])
plt.show()