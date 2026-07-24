import numpy as np
from . import timeit

class KMeans:
    @timeit
    def fit(self, X, k):
        # init centroids
        rng = np.random.default_rng()
        self._centroids = rng.choice(X, k)
        
        oldCenters = np.zeros(len(X))
        centers = np.zeros(len(X))
        while True:
            # new assignments
            counts = np.zeros(k)
            centroids = np.zeros_like(self._centroids)
            for i, x in enumerate(X):
                # distances
                distance = np.zeros(k)
                for j, c in enumerate(self._centroids):
                    distance[j] = np.sum((x - c)**2)
                
                # assignment
                m = np.argmin(distance)
                centers[i] = m 
                
                # count
                counts[m] += 1
                
                # accumulate centroids
                centroids[m] += x 
            
            # new centroids            
            self._centroids = centroids / counts[:,np.newaxis]
            
            # break condition
            if np.all(oldCenters == centers):
                break 
            else:
                oldCenters = centers
        return centers
                
    def __call__(self, X):
        distances = np.zeros((len(X), len(self._centroids)))
        for i, x in enumerate(X):
            for j, c in enumerate(self._centroids):
                distances[i,j] = np.sum((x - c)**2)
        return np.argmin(distances, axis=-1, keepdims=True)
