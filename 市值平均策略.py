import random
import numpy as np
import matplotlib.pyplot as plt

def trend(n): # 用于生成随机游走的行情走势
    # n表示n天的行情
    s = [1]
    for i in range(n):
        s.append((s[-1] * (1 + random.uniform(-0.002,0.002))))
    return s


def evaluation(asset): # 评估策略表现
    total_return = [1]
    annual_return = [1]
    for i in range(len(asset)):
        if i > 0:
            total_return.append((asset[i] / asset[i - 1]))
            a = total_return[-1] ** (365 / i)
            annual_return.append((a))
    return total_return,annual_return

def market_value_fix_invest(trend,base,amount,goal,n): # 策略，返回策略的总价值走势
    # base 本金
    # amount 每次基金增加市值
    # goal止盈收益率
    # n 预计投资时间
    asset = base.copy()
    fund_value = [0]
    fund_share = [0]
    for i in range(n):
        total_return,annual_return = evaluation(asset)
        if total_return[-1] < 1 + goal:
            if fund_value[-1] <= amount * (i + 1):
                if base[-1] >= amount * (i + 1) - fund_value[-1]:
                    fund_share.append((fund_share[-1] + (amount * (i + 1) - fund_value[-1]) / trend[i]))
                    base.append(((base[-1] - (amount * (i + 1) - fund_value[-1])) * (1 + 0.04) ** (1 / 365)))
                else:
                    fund_share.append((fund_share[-1] + base[-1] / trend[i]))
                    base.append((0))
            else:
                fund_share.append((fund_share[-1] - (fund_value[-1] - amount * (i + 1)) / trend[i]))
                base.append(((base[-1] + fund_value[-1] - amount * (i + 1)) * (1 + 0.04) ** (1 / 365)))
        elif i >= 365:
            fund_share.append((0))
            base.append((asset[-1]))
        fund_value.append((fund_share[-1] * trend[i]))
        asset.append((fund_value[-1] + base[-1]))
    return total_return,annual_return

s = trend(1000)
total_return,annual_return = market_value_fix_invest(s,[100000],100,0.3,500)
plt.plot(s)
plt.show()
plt.plot(total_return)
plt.show()
plt.plot(annual_return)
plt.show()


