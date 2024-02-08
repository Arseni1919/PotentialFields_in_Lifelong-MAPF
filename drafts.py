import numpy as np

a = ['a', 'b', 'c']
b = ['', 'baa', 'aaac']
c = np.intersect1d(a, b)
d = np.random.randn(10, 10)
print(len(c))