import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(arr)

arr[arr > 3] = 0

print(arr)