# 程序由模块构成，模块由语句构成，语句由表达式构成，表达式建立或处理对象
# python核心对象类型包括数字，字符串，列表，元组，字典，文件，集合，其他类型
# python是动态类型语言，同时也是强类型语言
print(2 ** 7)
from math import *
print(pi)
print(sqrt(25))
from random import *
print(random())
choice = choice([1,2,3,4])
print(choice)
s = '戴润是好学生'
print(s[:])
r = '个屁'
print(s + r)
# 字符具有不可变性
# s[0] = 'a'是错误的
print(s.find('戴'))
print(s.replace('好学生','聪明学生'))
line1 = 'aaa,bbb,ccc,ddd'
line2 = line1.split(',')
print(line2)
print(line2[0].upper())
print(line2[2].isalpha())
print(line2[2].isnumeric())
print('%s is not %s' % (line2[0],line2[0].upper()))
line2.append('bc')
print(line2)
line2.sort()
print(line2)
line2.reverse()
print(line2)
# 列表解析
m = [[1,2,3],[4,5,6],[7,8,9]]
print(m)
col2 = [row[1] for row in m]
print(col2)
col2 = [row[1] for row in m if row[1] % 2 == 0]
print(col2)
diag = [m[i][i] for i in range(3)]
print(diag)
doubles = [c * 2 for c in line2]
print(doubles)
G = (sum(row) for row in m)
print(next(G))
X = set('spam')
Y = {'h','a','m'}
print(X & Y)
print(X | Y)
print(X - Y)
x = {x ** 2 for x in [1,2,3,4]}
print(x)
y = {x ** 2 for x in (1,2,3,4)}
print(y)
from decimal import *
d = Decimal('2.11')
print(d)
print(getcontext().prec)
from fractions import Fraction
f = Fraction(2,3)
print(f)
L = [None] * 10
print(L)