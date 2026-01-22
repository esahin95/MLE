import numpy as np

class NaiveBayes:
    def fit(self, X, Y):
        # number of cases
        m, n = X.shape
        nClass1 = np.sum(Y)
        nClass0 = m - nClass1
        
        # estimate model parameters
        self._Py = nClass1 / m
        self._PI = np.zeros((2, n))
        for i in range(n):
            self._PI[1,i] = (1 + np.sum(X[Y.flat == 1,i])) / (nClass1 + 2)
            self._PI[0,i] = (1 + np.sum(X[Y.flat == 0,i])) / (nClass0 + 2)
            
    def prob(self, X):
        P = np.ones((X.shape[0], 2)) * np.array([[1-self._Py, self._Py]])
        for i in range(2):
            P[:,i:i+1] *= np.prod(self._PI[i:i+1]**X * (1 - self._PI[i:i+1])**(1 - X), axis=-1, keepdims=True)
        return P
    
    def predict(self, X):
        return np.argmax(self.prob(X), axis=-1, keepdims=True).astype(np.int8)
    
    def confusion(self, X, Y):
        P = self.predict(X)
        K = np.zeros((2,2))
        for p, y in zip(P, Y):
            K[p,y] += 1
        return K