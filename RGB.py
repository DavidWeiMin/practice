import numpy as np
b = [[0,0,0]]
set1 = {0,78,153,255}
for i in set1:
    for j in set1:
        for k in set1:
            a = [i,j,k]
            b = np.vstack((b,a))
print(b)
