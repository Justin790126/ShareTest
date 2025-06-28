import numpy as np
from layer1.layer1 import layer1_func

def layer2_func(arr):
    # Multiply by 2, then call layer1_func, then sum
    arr2 = arr * 2
    arr3 = layer1_func(arr2)
    return np.sum(arr3)