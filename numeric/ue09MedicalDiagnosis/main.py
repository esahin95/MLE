# imports
import sys
sys.path.append("..\\..\\")

import numpy as np

from mltools.data import DataCollection
from mltools.linear import NaiveBayes, NaiveBayesMixed

# load dataset
ds = DataCollection(fname="diagnosis.npz")
print(ds)

# training test split
rng = np.random.default_rng(45)
m = ds.X.shape[0]
idx = rng.permutation(m)
nTest = int(m * 0.2)
TTest, TTrain = ds.T[idx[:nTest]], ds.T[idx[nTest:]]
XTest, XTrain = ds.X[idx[:nTest]], ds.X[idx[nTest:]]
yTest, yTrain = ds.yInflam[idx[:nTest]], ds.yInflam[idx[nTest:]]

# train model
model = NaiveBayes()
model.fit(XTrain, yTrain)
print(model.confusion(XTest, yTest))

model = NaiveBayesMixed()
model.fit((XTrain, TTrain), yTrain)
print(model.confusion((XTest, TTest), yTest))