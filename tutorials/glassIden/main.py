# imports
import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

from mltools.linear import MultinomialLogistic

# load data
from ucimlrepo import fetch_ucirepo 
glass_identification = fetch_ucirepo(id=42) 
    
# data (as pandas dataframes) 
X = glass_identification.data.features.to_numpy()
y = glass_identification.data.targets.to_numpy()
    
# build model
model = MultinomialLogistic(X.shape[-1], np.max(y) + 1)
model.fit(X, y, alp=0.001, epochs=100000)

# evaluate model
C = model.confusion(X,y)
print(C)
print(np.sum(np.diagonal(C))/np.sum(C))