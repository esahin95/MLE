import numpy as np
from .Base import Base

class Linear(Base):
    def fit(self, X, y):
        # append bias
        X = self.append(X)
        
        # optimize objective
        self.weights, residuals, *_ = np.linalg.lstsq(X, y)
        print(residuals.item())