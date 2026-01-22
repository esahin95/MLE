import numpy as np
from .NaiveBayes import NaiveBayes

class NaiveBayesMixed(NaiveBayes):
    def fit(self, X, Y):
        Xnom, Xcar = X
        
        # probabilities for nominal scales
        super().fit(Xnom, Y)
        
        # gauss distribution for cardinal scales
        self._mu = np.zeros((2, Xcar.shape[-1]))
        self._std = np.zeros_like(self._mu)
        for i in range(2):
            self._mu[i] = np.mean(Xcar[Y==i])
            self._std[i] = np.std(Xcar[Y==i])
            
    def prob(self, X):
        Xnom, Xcar = X 
        
        P = super().prob(Xnom)
        P *= np.exp(-0.5*(Xcar - self._mu.T)**2/self._std.T**2) / (self._std.T*np.sqrt(2*np.pi))
        return P