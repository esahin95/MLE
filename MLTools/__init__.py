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


# cropped image
kwfig = {
    'format':'pdf',
    'bbox_inches':'tight',
    'pad_inches':0.0,
    'transparent':True
}


from . import data
from . import ode
from . import ann
from . import cluster
from . import factor
from . import linear
from . import svm
from . import utils