import numpy as np

num = 64
label_vector = np.zeros((num, 10), dtype=np.float)
for i in range(0, num):
    label_vector[i, i/8] = 1.0

print label_vector