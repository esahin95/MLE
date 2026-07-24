import numpy as np


class Optimizer:
    def fit(self, f, x0, *, maxIter, eps):
        raise NotImplementedError()