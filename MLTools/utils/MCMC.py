import numpy as np
from . import timeit

class MCMC:
    def __init__(self, state, sigma=1.0, seed=0):
        self._rng = np.random.default_rng(seed)
        self._sigma = sigma
        self.state = state
        
    def step(self, f):
        # proposal
        q = self._rng.normal(loc=self.state, scale=self._sigma)
        
        # acceptance
        accept = self._rng.random() < min(1.0, f(q)/f(self.state))
        if accept:
            self.state = q
        return accept
    
    @timeit
    def run(self, f, n, burnin=0, lag=1):
        # burnin period
        for _ in range(burnin):
            self.step(f)

        # collect samples
        D = np.zeros(n)
        nAccept = 0
        for i in range(n):
            for j in range(lag):
                nAccept += self.step(f)
            D[i] = self.state
        
        print(f'Acceptance probability: {nAccept/(n*lag):.4f}')
        return D