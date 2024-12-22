# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:43:41 2024

@author: sahin
"""
import numpy as np
    
class DataCollection:
    def __init__(self, **kwargs):
        ''' 
        Initialize from keyword arguments
        '''
        self.__dict__.update(**kwargs)
        
    def load(self, fname):
        ''' 
        Set from numpy compressed file
        '''
        npz = np.load(fname)
        self.__dict__.update(npz)
    
    def save(self, fname='Data/tmp.npz'):
        ''' 
        Save contents to file
        '''
        np.savez(fname, **self.__dict__)
        
    def __repr__(self):
        ''' 
        Representation of data collection
        '''
        return f'{self.__class__!s} containing {self.__dict__.keys()!r}'