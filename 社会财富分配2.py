import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

class Person:
    def __init__(self,value):
        self.asset = [100,]
    
    def bankrupt(self):
        if self.asset[-1] <= 0:
            return True
        else:
            return False

class fuerdai(Person):
    
    