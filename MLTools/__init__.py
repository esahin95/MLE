# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:41:57 2024

@author: sahin
"""

# import namespaces
from MLTools.dataset import DataCollection
from MLTools.linearModels import LinearRegression, LogisticRegression, Perceptron, MultinomialLogisticRegression, RidgeRegression
from MLTools.neuralNetworks import MLP
from MLTools.ode import FreeFall, Pendulum, PopDynamics
from MLTools.MCMC import MCMC
from MLTools.cluster import KMeans
from MLTools.factorization import PCA
from MLTools.probabilistic import NaiveBayes, NaiveBayesMixed, ICA
from MLTools.optim import NealderMead, GN
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
    "SVC"
]