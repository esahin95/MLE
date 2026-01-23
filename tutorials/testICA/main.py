# imports
import sys
sys.path.append("..\\..\\")

import numpy as np
import matplotlib.pyplot as plt

from mltools.data import DataCollection
from mltools.factor import ICA

# load data
ds = DataCollection("ica.npz")

# build model
model = ICA()
model.fit(ds.X, lr=5e-4, bs=100, epochs=10)

# post process    
fig, axs = plt.subplots(16,16,figsize=(10,10),subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw={'hspace': 0, 'wspace': 0})
for i, ax in enumerate(axs.flat):
    ax.imshow(model[i].reshape(16,16), cmap = 'gray')
plt.tight_layout()
plt.show()