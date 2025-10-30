# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:59:52 2025

@author: sahin
"""

import numpy as np

class MCMC:
    def __init__(self, state, sigma, seed=0):
        self._rng = np.random.default_rng(seed)
        self._sigma = sigma
        self.state = state
        
    def step(self, fun):
        # proposal
        q = self._rng.normal(loc=self.state, scale=self._sigma)
        
        # acceptance
        accept = self._rng.random() < min(1.0, fun(q)/fun(self.state))
        if accept:
            self.state = q
        return accept