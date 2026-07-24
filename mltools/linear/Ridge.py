import numpy as np
from .Base import Base

class Ridge(Base):
    def fit(self, X, y, lam=0.01):        
        # append bias 
        X = self.append(X)
        
        # append penalty
        d = X.shape[1]
        X = np.vstack((X, np.sqrt(lam)*np.eye(d)))
        y = np.vstack((y, np.zeros((d, 1))))
        
        # solve least squares
        self.weights, residuals, *_ = np.linalg.lstsq(X, y)
        print(residuals.item())