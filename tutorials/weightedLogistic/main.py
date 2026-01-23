# imports
import sys
sys.path.append("..\\..\\")

import numpy as np
import matplotlib.pyplot as plt

from mltools.linear import WeightedLogistic
from mltools.data import DataCollection

# load data
ds = DataCollection(fname='wLogistic.npz')

# train model    
model = WeightedLogistic(nFeatures=ds.X.shape[-1])
model.fit(ds.X, ds.y, tau=0.1)

# test model   
yp = model(ds.Xt)

# post processing
fig, axs = plt.subplots(1, 2, figsize=(6,3))
P = ds.y.flatten() > 0.0
N = ds.y.flatten() < 0.0
for i, y in enumerate([ds.yt, yp]):
    axs[i].contourf(
        ds.Xt[:,0].reshape(ds.sz), 
        ds.Xt[:,1].reshape(ds.sz), 
        y.reshape(ds.sz), 
        cmap='coolwarm', 
        alpha=0.5, 
        vmin=-1.0, vmax=1.0
    )
    axs[i].scatter(ds.X[P,0], ds.X[P,1], color='r')
    axs[i].scatter(ds.X[N,0], ds.X[N,1], color='b')
plt.show()