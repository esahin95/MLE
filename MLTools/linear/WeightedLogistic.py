import numpy as np
from .Logistic import Logistic

class WeightedLogistic(Logistic):    
    def fit(self, X, y, tau, lam=1e-4):
        self._X = self.append(X)
        self._y = y.copy()
        self._scl = 2.0 * tau**2
        self._lam = lam
        self._lamI = lam * np.eye(self._X.shape[-1])
        
    def predict(self, x, maxIter=50):
        for i in range(maxIter):
            # distances
            d = np.exp(-np.sum((self._X - x)**2, axis=-1, keepdims=True) / self._scl)
            
            # logistics
            g = self.logit(self._y * (self._X @ self.weights))
            
            # gradient
            gradL = self._lam * self.weights - self._X.T @ (d * (1 - g) * self._y)
            
            # Hessian
            HL = self._lamI + self._X.T @ ((d * g * (1-g)) * self._X)
            
            # direction
            d = np.linalg.solve(HL, -gradL)
            
            # update
            self.weights += d 
            
            # termination condition
            if np.linalg.norm(gradL) < 1e-5:
                return super().__call__(x[:,1:])[0,0]
        raise Exception('Newton did not converge')
        
    def __call__(self, X):
        # append bias
        X = self.append(X)
        
        # predict for each x
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            self.weights *= 0.0
            y[i] = self.predict(X[i:i+1])
        return y