# imports
import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

from mltools.cluster import KMeans
from mltools import kwfig

# Load image
img = plt.imread("monarch.png")
img = img[:,:,:3]
h, w, _ = img.shape

# Clustering in color space
model = KMeans()
X = img.reshape(-1,3)
model.fit(X, 3)
c = model(X)

# Set color to centroids
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    Z[i] = model._centroids[c[i]]
newImg = Z.reshape(img.shape)

# Compare images
wspace = 0.02
fig, axs = plt.subplots(1, 2, figsize=(2*(w/h)*5/(1-wspace), 5))
for ax, I in zip(axs, [img, newImg]):
    ax.imshow(I)
    ax.axis('off')
    ax.set_aspect('equal')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=wspace)
plt.savefig('monarch.pdf', **kwfig)