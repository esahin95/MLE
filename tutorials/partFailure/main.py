# imports
import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

from mltools.linear import Logistic, Perceptron
from mltools.data import DataCollection

test = "Logistic"
    
# dataset from MLE book
data = DataCollection(fname="partFailure.npz")
print(data.X, data.y, sep='\n')

# build model
match test:
    case "Logistic":
        model = Logistic(1)
        model.fit(data.X, data.y, alp=1.0, epochs=10)
        
    case "Perceptron":
        model = Perceptron(1)
        model.fit(data.X, data.y, alp=0.1, epochs=300)
        
    case _:
        raise ValueError("unknown model type")
print(model)

# compare prediction
fig, ax = plt.subplots(1, 1, figsize=(4,2))
ax.scatter(data.X, data.y, c='b')
ax.scatter(data.X, model(data.X), facecolors='none', edgecolors='r')
ax.set(xlabel='$T$', ylabel='Failure')
plt.show()