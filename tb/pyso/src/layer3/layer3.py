import numpy as np
from layer2.layer2 import layer2_func

def layer3_func(arr):
    # Apply a sine, then call layer2_func, then take log
    arr2 = np.sin(arr)
    val = layer2_func(arr2)
    return np.log(val + 1)