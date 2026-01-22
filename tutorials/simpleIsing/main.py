# imports
import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

# mesh
N = 50

# initial state
rng = np.random.default_rng(0)
L = np.ones((N,N))
L[rng.random((N,N)) < 0.75] = -1 

# total energy
def energy(L):
    return np.sum(- L * (np.roll(L,1,axis=0) + np.roll(L,1,axis=1)))

# params
times = 300000
BJ = 0.1
        
# run metropolis
M = np.repeat(L.sum(), times)
E = np.repeat(energy(L), times)
for t in range(1, times):
    # proposal step
    x = rng.integers(N)
    y = rng.integers(N)
    
    # compute energy change
    Ei = -L[x,y] * (L[(x+1) % N, y] + L[(x-1) % N, y] + L[x, (y+1) % N] + L[x, (y-1) % N])
    dE = -2.0 * Ei
    
    # acceptance step
    accept = np.random.random() < min(1, np.exp(-BJ*dE))
    if accept:
        L[x,y] *= -1 
    
    # macroscopic observables
    if accept:
        E[t] = E[t-1] + dE 
        M[t] = M[t-1] + L[x,y] * 2
    else:
        E[t] = E[t-1]
        M[t] = M[t-1]
print(f'Energy: {E[times//2:].mean()}, Magnetization: {M[times//2:].mean()/N**2}')
    
# postprocessing
fig, axs = plt.subplots(1,3,figsize=(12,4))
axs[0].plot(E)
axs[1].plot(M / N**2)
axs[2].imshow(L)
plt.show()