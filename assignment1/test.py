import numpy as np

a = np.array([[[1, 2], [3, 4], [5, 6]]])
b = np.array([7, 8])
a = np.delete(a, 0, axis=1)
print(a)
a = np.insert(a, 2, b, axis=1)
print(a)
