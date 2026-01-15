# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:25:18 2025

@author: sahin
"""

import numpy as np
from scipy.spatial import KDTree
from . import timeit

class KMeans:
    def fit(self, X, k):
        # init centroids
        rng = np.random.default_rng()
        self._centroids = rng.choice(X, k)
        
        oldCenters = np.zeros(len(X))
        while True:
            # new assignments
            centers = np.zeros(len(X))
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
                
    def __call__(self, X):
        distances = np.zeros((len(X), len(self._centroids)))
        for i, x in enumerate(X):
            for j, c in enumerate(self._centroids):
                distances[i,j] = np.sum((x - c)**2)
        return np.argmin(distances, axis=-1, keepdims=True)
    
    
class DBSCAN:
    @timeit    
    def fit(self, X, minPts, r, leafsize=30):        
        # Processing neighbourhoods
        kdTree = KDTree(X, leafsize=leafsize)
        
        # Initialize variables
        m = X.shape[0]
        visited = np.zeros(m, dtype=bool)
        cluster = np.zeros(m, dtype=int)
        C = 0
        
        # Loop over dataset
        for i in range(m):
            if visited[i]:
                continue
            
            # Index set of neighbours
            visited[i] = True 
            N = set(kdTree.query_ball_point(X[i], r))
            if len(N) < minPts:
                continue
            
            # Generate new cluster
            C += 1
            cluster[i] = C
            while len(N) > 0:
                j = N.pop()
                
                # Add to cluster if possible
                if cluster[j] > 0:
                    continue
                cluster[j] = C 
                
                # Expand neighbourhood
                visited[j] = True
                NExp = set(kdTree.query_ball_point(X[j], r))
                if len(NExp) >= minPts:
                    N = N | NExp
                            
        # Return clustering
        return cluster.reshape(-1,1)