# imports
import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from mltools.utils import MCMC

# data
p = 0.3 
m = 5
rng = np.random.default_rng(0)
X = rng.random(m) < p

# prior probability density
a, b = 2, 2
def prior(x):
    if x >= 1.0 or x <= 0.0:
        return 0.0
    else:
        return x**(a-1) * (1-x)**(b-1)

# likelihood
h = np.sum(X)
def like(x):
    return x**h * (1-x)**(m-h)

# stationary distribution
def f(x):
    return like(x) * prior(x)

# build Markov chain
mc = MCMC(0.5, sigma=1e-1)
D = mc.run(f, 20000, burnin=100, lag=1)

# postprocessing
fig, axs = plt.subplots(1,2,figsize=(12,4))
axs[0].hist(D, bins=50, density=True)
Z = np.linspace(0,1,1000)
axs[0].plot(Z, beta.pdf(Z, 2, 2), 'b')
axs[0].plot(Z, beta.pdf(Z, a+h, b+m-h), 'r')
axs[1].plot(D, range(len(D)))
plt.show()