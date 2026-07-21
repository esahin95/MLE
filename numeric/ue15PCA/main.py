# imports
import sys
sys.path.append("..\\..\\")

import numpy as np
import matplotlib.pyplot as plt

from mltools.data import DataCollection
from mltools.factor import PCA 

# load data
ds = DataCollection("pca.npz")

# build model
model = PCA()
model.fit(ds.X)

# post process
fig, axs = plt.subplots(16,16,figsize=(10,10),subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw={'hspace': 0, 'wspace': 0})
for i, ax in enumerate(axs.flat):
    ax.imshow(model[i][1].reshape(16,16), cmap = 'gray')
plt.tight_layout()
plt.show()