import matplotlib.pyplot as plt

from mltools.linear import Logistic, Perceptron
from mltools.data import DataCollection

test = "Logistic"

# dataset from MLE book
ds = DataCollection(fname="partFailure")

# build model
match test:
    case 'Logistic':
        model = Logistic(1)
        model.fit(ds.X, ds.y, alp=1.0, epochs=10)

    case 'Perceptron':
        model = Perceptron(1)
        model.fit(ds.X, ds.y, alp=0.1, epochs=300)

    case _:
        raise ValueError("unknown model type")

# compare prediction
fig, ax = plt.subplots(1, 1, figsize=(4,2))
ax.scatter(ds.X, ds.y, c='b')
ax.scatter(ds.X, model(ds.X), facecolors='none', edgecolors='r')
ax.set(xlabel='$T$', ylabel='Failure')
plt.show()