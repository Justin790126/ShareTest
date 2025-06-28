import numpy as np
import libtest

# Try hack to get pyc, but failed
import sys
sys.dont_write_bytecode = True

arr = np.arange(5, dtype=np.float64)
result = libtest.process(arr)
print("Result from libtest.process:", result)