
# Not allow to generate .pyc file
import sys
sys.dont_write_bytecode = True



# distutils: language = c
# cython: language_level=3

import numpy as np
cimport numpy as np

from layer3.layer3 import layer3_func

def process(np.ndarray[np.float64_t, ndim=1] arr):
    '''
    Accept a float64 numpy array and process using nested layers.
    '''
    return layer3_func(arr)