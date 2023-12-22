import numpy as np

a = ['a', 'b', 'c']
b = ['', 'baa', 'aaac']
c = np.intersect1d(a, b)
print(len(c))