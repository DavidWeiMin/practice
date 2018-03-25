# -*- coding: utf-8 -*-
# 设置回测区间
# 设置参照基准
# 设置本金
# 启动Wind API
from WindPy import *
w.start()
# 登陆账户
LogonID=w.tlogon('0000',0,'WS7794102301','1395373646','sh')
# 获取数据
# 设置策略类型
# 设置买卖条件
w.torder('600000.SH', 'buy', 9.8, 100,logonid=1) 
# 
#