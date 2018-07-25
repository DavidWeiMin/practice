import math
import matplotlib.pyplot as plt

def combine(n,i):
    if n == 0 or i ==0 or n == i:
        return 1
    else:
        a = set(range(1,n + 1))
        b = set(range(1,i + 1))
        c = a - b
        c = list(c)
        result = 1
        for j in range(1,n - i + 1):
            result = result * c[j - 1] / j
        return result

p = []
for n in range(10,30):
    probability = 0
    level = 0.6
    pass_level = math.floor(n * level) + 1
    for i in range(pass_level,n + 1):
        probability += combine(n,i) * 0.25 ** i * 0.75 ** (n - i)
    p.append((probability))
plt.plot(range(10,30),p)
plt.show()