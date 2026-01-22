# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:25:41 2025

@author: sahin
"""

import numpy as np

class PCA:
    def __getitem__(self, idx):
        return self._lam[idx], self._V[idx]
        
    def fit(self, X, k=None):
        # covariance matrix
        K = X @ X.T / X.shape[1]
        
        # matrix factorization
        [lam, V] = np.linalg.eigh(K)
        
        # save reduced basis in reverse order
        if k is None:
            self._lam, self._V = lam[::-1], V[:,::-1].T 
        else:
            self._lam, self._V = lam[:-k-1:-1], V[:,:-k-1:-1].T