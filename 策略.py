import random
import numpy as np
import matplotlib.pyplot as plt

def trend(n): # 用于生成随机游走的行情走势
    # n表示n天的行情
    s = [1]
    for i in range(n):
        s.append((s[-1] * (1 + random.uniform(-0.01,0.0104))))
    return s

def market_value_fix_invest(trend,base,amount,goal,n): # 策略，返回策略的总价值走势
    # base 本金
    # amount 每次基金增加市值
    # goal止盈收益率
    # n 预计投资时间
    fund_value = [0]
    fund_share = [0]
    return_rate = [0]
    withdraw = [0]
    t = 0
    withdraw_return = 0
    earnings = [0]
    for i in range(n):         
        if i < 1 or (earnings[-1] - withdraw_return) / (base[0] - min(base)) < goal:
            if fund_value[-1] <= amount * (i + 1 - t):
                if base[-1] >= amount * (i + 1 - t) - fund_value[-1]:
                    fund_share.append((fund_share[-1] + (amount * (i + 1 - t) - fund_value[-1]) / trend[i]))
                    base.append(((base[-1] - (amount * (i + 1 - t) - fund_value[-1]))))
                else:
                    fund_share.append((fund_share[-1] + base[-1] / trend[i]))
                    base.append((0))
            else:
                fund_share.append((fund_share[-1] - (fund_value[-1] - amount * (i + 1 - t)) / trend[i]))
                base.append(((base[-1] + fund_value[-1] - amount * (i + 1 - t))))
        else:
            withdraw.append((i))
            fund_share.append((0))
            base.append((base[-1] + fund_value[-1]))
            t = i
        fund_value.append((fund_share[-1] * trend[i]))
        earnings.append((fund_value[-1] + base[-1] - base[0]))
        return_rate.append((earnings[-1] / (base[0] - min(base))))
        # if fund_share[-1] > 0:
        #     # cost.append((fund_value[-1] / fund_share[-1]))
        # else:
            # cost.append((cost[-1]))        
        if t == i:
            withdraw_return = earnings[-1]
    return_rate = return_rate + np.ones_like(return_rate)
    plt.plot(trend)
    plt.plot(return_rate)
    plt.legend(['market','strategy'])
    plt.show()
    # plt.plot(base[0] * np.ones_like(base) - base)
    # plt.scatter(withdraw,np.zeros_like(withdraw))
    # plt.show()
    # plt.plot(fund_value+base)
    # plt.show()
    # plt.plot(earnings)
    # plt.show()
    return return_rate,withdraw

s = trend(1600)
return_rate,withdraw = market_value_fix_invest(s,[100000],100,0.15,1598)
plt.show()
