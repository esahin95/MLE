# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:41:57 2024

@author: sahin
"""

# timing decorator
from timeit import default_timer

def timeit(f):
    def timed(*args, **kwargs):
        ts = default_timer()
        rs = f(*args, **kwargs)
        te = default_timer()
        print(f'func:{f.__name__!r} took: {te-ts:.6e} s')
        return rs
    return timed

# import namespaces
from MLTools.dataset import DataCollection
from MLTools.linearModels import LinearRegression, LogisticRegression, Perceptron, MultinomialLogisticRegression, RidgeRegression
from MLTools.neuralNetworks import MLP
from MLTools.ode import FreeFall, Pendulum, PopDynamics
from MLTools.MCMC import MCMC
from MLTools.cluster import KMeans
from MLTools.factorization import PCA
from MLTools.probabilistic import NaiveBayes, NaiveBayesMixed, ICA
from MLTools.optim import NealderMead, GN, LogBarrier
from MLTools.svm import SVC

# visible classes in external applications
__all__ = [
    "DataCollection",
    "LinearRegression",
    "LogisticRegression",    
    "Perceptron",
    "MultinomialLogisticRegression",
    "RidgeRegression",
    "Pendulum",
    "FreeFall",
    "PopDynamics",
    "MLP",
    "MCMC",
    "KMeans",
    "PCA",
    "NaiveBayes",
    "NaiveBayesMixed",
    "ICA",
    "NealderMead",
    "GN",
    "LogBarrier",
    "SVC",
    "timeit"
]