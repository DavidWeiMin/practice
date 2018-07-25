import math
from random import *

import matplotlib.pyplot as plt
import numpy as np


class Strategy:
    num_strategy = 0
    def __init__(self,period=756):
        Strategy.num_strategy += 1
        self.period = period
        self.volatility = [0]
        self.volatility_freq = [0]
        self.market = [1]
        self.volatility = np.random.normal(0.0001,0.015,period)
        if max(abs(self.volatility)) > 0.1:
            self.volatility = self.volatility / max(abs(self.volatility)) * 0.1
        for i in self.volatility:
            self.market.append((self.market[-1] * (1 + i)))

    def start_strategy(self,start_date,end_date,frequence=5):
        self.frequence = frequence
        self.capital = [10000000000]
        self.cost = []
        self.share = [0]
        for i in range(start_date - 1,end_date):
            if i % self.frequence == 0:
                self.share.append((self.share[-1] + 10 / self.market[i]))
                self.cost.append((10 * (len(self.share) - 1) / self.share[-1]))
                self.capital.append((self.capital[-1] - 10))
        self.market_freq = [self.market[i] for i in range(start_date - 1,end_date) if i % self.frequence == 0]
        self.strategy_return = np.array(self.market_freq) / np.array(self.cost)

    def plot_market(self,start_date,end_date,frequence=1):
        self.market_freq = [self.market[i] for i in range(start_date - 1,end_date) if i % frequence == 0]
        plt.plot(self.market_freq)
        plt.show()

    def plot_compare(self,start_date,end_date):
        self.market_freq = [self.market[i] for i in range(start_date - 1,end_date) if i % self.frequence == 0]
        plt.plot(self.cost)
        plt.plot(self.market_freq)
        plt.plot(np.array(self.market_freq) / np.array(self.cost))
        plt.legend(['strategy','market','alpha'])
        plt.show()

    def plot_volatility(self,frequence=1):
        self.volatility_freq = [self.volatility[i] for i in range(self.period) if i % frequence == 0]
        plt.hist(self.volatility_freq,bins=round(math.sqrt(self.period)))
        plt.show()

if __name__=='__main__':
    period = 10000
    s1 = Strategy(period)
    s1.plot_market(1,period,50)
    # s1.start_strategy(1,756,3)
    # s1.plot_strategy()
    # s1.plot_compare(1,756)
    # s1.plot_volatility()
