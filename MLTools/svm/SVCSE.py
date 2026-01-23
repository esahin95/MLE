import numpy as np
from .SVC import SVC

class SVCSE(SVC):
    def __init__(self, sigma=1.0):
        self._sig2 = sigma**2
        
    def K(self, x, z):
        return np.exp(-0.5*np.linalg.norm(x-z)**2 / self._sig2)