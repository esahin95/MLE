import numpy as np
    
class DataCollection:
    def __init__(self, fname=None, **kwargs):
        self.__dict__.update(**kwargs)
        if fname:
            self.load(fname)
        
    def load(self, fname):
        npz = np.load(fname)
        self.__dict__.update(npz)
    
    def save(self, fname='Data/tmp.npz'):
        np.savez(fname, **self.__dict__)
        
    def __repr__(self):
        return f'{self.__class__!s} containing {self.__dict__.keys()!r}'