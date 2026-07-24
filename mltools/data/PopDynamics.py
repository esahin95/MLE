from .DataCollection import DataCollection
from ..ode import PopDynamics as ODE
import numpy as np

class PopDynamics(DataCollection):
    def __init__(self, *, fname='popDynamics', n=10, b=2.0, r=3.0, K=15.0):
        ode = ODE(r, K)

        X = np.arange(1, n+1).reshape(-1,1) * 5.0 / n
        y = ode.integrate(b, 1e-2, T=X)

        XTest = np.linspace(0.0, 5.0, 100).reshape(-1,1)
        yTest = ode.integrate(b, 1e-2, T=XTest)

        super().__init__(fname, 'tmp', X=X, y=y, XTest=XTest, yTest=yTest)