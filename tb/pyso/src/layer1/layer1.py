import numpy as np

def layer1_func(arr):
    # Add 1 to every element, then square it
    result = np.square(arr + 1)
    return result