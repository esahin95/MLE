import numpy as np
from importlib.resources import files

class DataCollection:
    def __init__(self, fname='tmp', dname='data', format='npz', **kwargs):
        self.path = files(dname) / (fname + '.' + format)

        if kwargs:
            self.__dict__.update(**kwargs)
            self.save()
        else:
            self.load()

    def load(self):
        npz = np.load(self.path)
        self.__dict__.update(npz)

    def save(self):
        np.savez(self.path, **self.__dict__)

    def __repr__(self):
        return f'{self.__class__!s} containing {self.__dict__.keys()!r}'