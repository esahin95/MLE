import numpy as np

import mltools.data.FreeFall as Data
import mltools.linear.Linear as Model

ds = Data()
model = Model(nFeatures=1)

# train model
model.fit(ds.X, ds.y)
print(model)

# test on moon
g, h, v = 1.62, 1.5, 0.1
x = v / np.sqrt(h * g)
tTrue = np.sqrt(h / g) * (x + np.sqrt(x**2 + 2.0))
tPred = np.sqrt(h / g) * model(x.reshape(1,1))
print(tTrue, tPred)