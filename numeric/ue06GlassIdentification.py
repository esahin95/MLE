import numpy as np

from mltools.data import DataCollection
from mltools.linear import MultinomialLogistic

# load data
ds = DataCollection(fname="glass")

# build model
model = MultinomialLogistic(ds.X.shape[-1], np.max(ds.y) + 1)
model.fit(ds.X, ds.y, alp=0.002, epochs=200000)

# evaluate model
C = model.confusion(ds.X,ds.y)
print('Confusion matrix:', C, sep='\n')
print('Precision: ', np.sum(np.diagonal(C))/np.sum(C))