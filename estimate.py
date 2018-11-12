import numpy as np
from random import random
import sympy
import math
def calculation(alpha,beta,gamma,m):
    # m = 10 # 精度
    # alpha = np.ones(m + 1)
    # beta = np.ones(m + 1)
    a = np.zeros((m + 1,m + 1))
    b = np.zeros((m + 1,m + 1))
    i = j = 0
    shift = True
    a[i,i] = alpha[i]
    print('a',i + 1,i + 1)
    print(a[i,i])
    while True:
        if i == 0 and shift:
            j += 1
            if j > m - 1:
                break
            i = j
            a[i,i] = gamma[i]
            shift = True
            print('a',i + 1,i + 1)
            print(a[i,i])
        else:
            if shift:
                i -= 1
                b[i,j] = np.dot(alpha[:(m - i - 1)],a[(i + 1):m,m - 1])
                shift = False
                print('b',i + 1,j + 1)
                print(b[i,j])
            else:
                temp = [b[x,m - i + x - 1] for x in range(i + 1)]
                a[i,j] = np.dot(beta[i::-1],temp)
                shift = True
                print('a',i + 1,j + 1)
                print(a[i,j])
    return a,b

def get_alpha(m):
    x = sympy.symbols('x')
    f = (sympy.exp(-x) + 0.5 * sympy.exp(-0.5 * x)) / 2
    alpha = [f]
    while len(alpha) < m:
        alpha.append(sympy.diff(alpha[-1]))
    alpha = [i.subs(x,0) for i in alpha]
    return alpha

def get_beta(m):
    t = sympy.symbols('t')
    f = 1 / (1 - t)
    beta = [f]
    while len(beta) < m:
        beta.append(sympy.diff(beta[-1]))
    beta = [j.subs(t,0) / math.factorial(i) for i,j in enumerate(beta)]
    return beta

def get_gamma(m):
    return get_beta(m)
m = 10
a,b = calculation(get_alpha(m + 1),get_beta(m + 1),get_gamma(m),m)
# w = b[0,:].sum()
# print(w)