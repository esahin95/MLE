import numpy as np

class ICA:
    def __getitem__(self, idx):
        return self._W[idx]
    
    def fit(self, X, bs=100, lr=5e-4, epochs=10):
        n, m = X.shape
        rng = np.random.default_rng()
        
        self._W = np.eye(n)
        for epoch in range(epochs):
            I = rng.permutation(range(m))
            for j in range(m//bs):
                Xbatch = X[:,I[j*bs:j*bs+bs]]
                Gbatch = 1.0 - 2.0 / (1.0 + np.exp(- self._W @ Xbatch))
                self._W += lr * (Gbatch @ Xbatch.T + bs * np.linalg.inv(self._W.T))
                
        return self._W @ X