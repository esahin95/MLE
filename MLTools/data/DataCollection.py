import numpy as np
import os
from importlib.resources import files

class DataCollection:
    def __init__(self, fname=None, dname=None, **kwargs):
        self.__dict__.update(**kwargs)
        if fname:
            self.load(fname, dname)

    def load(self, fname, dname=None):
        if dname is None:
            fullName = files('data') / (fname + '.npz')
        else:
            fullName = os.path.join(dname, fname + '.npz')
        npz = np.load(fullName)
        self.__dict__.update(npz)

    def save(self, fname):
        np.savez(os.path.join('tmp', fname), **self.__dict__)

    def __repr__(self):
        return f'{self.__class__!s} containing {self.__dict__.keys()!r}'