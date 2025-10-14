# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:59:52 2025

@author: sahin
"""

import numpy as np

class MCMC:
    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)
        
    def step(self, fun):
        # proposal
        q = self.prop()
        
        # acceptance
        if self._rng.random() < min(1.0, fun(q)/fun(self._state)):
            self._state = q
            return 1
        else:
            return 0
            
    def prop(self):
        return self._rng.normal(loc=self._state, scale=self._sigma)
    
    def run(self, x, fun, n=1, burnin=0, lag=1, sigma=1.0):
        # initialize
        self._state = x
        self._sigma = sigma
        
        # burnin period
        nAccept = 0
        for _ in range(burnin):
            nAccept += self.step(fun)
            
        # collect samples
        D = np.zeros(n)
        for i in range(n):
            for j in range(lag):
                nAccept += self.step(fun)
            D[i] = self._state
        print(f'Acceptance probability: {nAccept/(n*lag+burnin):.4f}')
        return D