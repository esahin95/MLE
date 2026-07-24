import numpy as np
from scipy.spatial import KDTree
from . import timeit

class DBSCAN:
    @timeit    
    def fit(self, X, minPts, r, leafsize=30):        
        # Processing neighbourhoods
        kdTree = KDTree(X, leafsize=leafsize)
        
        # Initialize variables
        m = X.shape[0]
        cluster = np.zeros(m, dtype=int)
        C = 0
        
        # Loop over dataset
        for i in range(m):
            if cluster[i] > 0:
                continue
            
            # Index set of neighbours
            N = set(kdTree.query_ball_point(X[i], r))
            if len(N) < minPts:
                continue
            
            # Generate new cluster
            C += 1
            cluster[i] = C
            while len(N) > 0:
                j = N.pop()
                
                # Add point to cluster or ignore
                if cluster[j] > 0:
                    continue
                cluster[j] = C 
                
                # Expand neighbourhood
                NExp = set(kdTree.query_ball_point(X[j], r))
                if len(NExp) >= minPts:
                    N = N | NExp
                            
        # Return clustering
        return cluster.reshape(-1,1)