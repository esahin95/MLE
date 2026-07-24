import numpy as np
from .Base import Base
from . import timeit

class Lasso(Base):
    @timeit
    def fit(self, X, y, lam=0.01, epochs=50):
        # append bias
        X = self.append(X)
        
        # optimization loop
        for epoch in range(epochs):
            dw = 0
            # coordinate ascent
            for i in range(X.shape[-1]):                
                # eliminate coordinate
                w, self.weights[i,0] = self.weights[i,0], 0
                
                # reduced costs 
                J = y - X @ self.weights
                a = np.sum(X[:,i]**2)
                b = (X[:,i] @ J)[0]
                
                # potential solutions
                wip = max(0, (b - lam) / a)
                win = min(0, (b + lam) / a)
                
                # optimal solution
                Jp = 0.5*a*wip**2 - (b - lam)*wip
                Jn = 0.5*a*win**2 - (b + lam)*win
                if Jp < Jn:
                    self.weights[i] = wip
                else:
                    self.weights[i] = win
                    
                # maximum change
                dw = max(dw, np.abs(w - self.weights[i,0]))
                
            # current loss
            if dw < 1e-5:
                L = 0.5 * np.sum((X @ self.weights - y)**2) + lam * np.linalg.norm(self.weights, 1)
                print(f'terminated at iteration {epoch} with residual {L}')
                return
        print('Lasso failed to converge')