# s = 'string'
# s = s + '123' # 字符串连接 'string123'
# s = s * 3 # 字符串复制 'string123string123string123'
# s = 'a' + s[1:] # 'atring123string123string123'
# s.find('tr') # 1
# s.replace('g','G') # 'atrinG123strinG123strinG123'
# line = 'a,b,C,D'
# line.split(',') # ['a', 'b', 'C', 'D']
# line.upper() # 'A,B,C,D'
# line[0] # 'a'
# line.lower() # 'a,b,c,d'
# line[4] # 'C'
# dir(s) # 返回字符串对象的所有属性方法
# help(s.split) # 查询如何使用split方法
# L1 = [123,'qwer',12.3]
# L1[0] # 123
# L1[-1] # 12.3
# L1[-2] # 'qwer'
# L2 = [2,1,3]
# L3 = L1 + L2 # [123, 'qwer', 12.3, 2, 1, 3]
# L3.append(('zxcv')) # [123, 'qwer', 12.3, 2, 1, 3, 'zxcv']
# L1.append(([1,2,33])) # [123, 'qwer', 12.3, [1, 2, 33]]
# L3.pop((5)) # 返回删除值3，删除后L3=[123, 'qwer', 12.3, 2, 1, 'zxcv']
# L2.sort() # [1,2,3]
# L3.reverse() # [3, 2, 1]
# L3.remove('zxcv') # [123, 'qwer', 12.3, 1, 2]
# L3.insert(5,'insert') # [123, 'qwer', 12.3, 1, 2, 'insert']
# L4 = [[1,2,3],[2,3,4],[3,4,5]]
# L4[1][2] # 4
# L4[1,2] # 错误
# col2 = [row[1] for row in L4] # [2,3,4]
# L5 = [row[1] + 1 for row in L4] # [3,4,5]
# L6 = [row[1] for row in L4 if row[1] % 2 == 0] # [2,4]
# diag = [L4[i][i] for i in [0,1,2]] # [1, 3, 5]
# doubles = [c * 2 for c in 'student'] # ['ss', 'tt', 'uu', 'dd', 'ee', 'nn', 'tt']

import shelve

class Person:
    def __init__(self,name,job=None,pay=0):
        self.name = name
        self.job = job
        self.pay = pay
    def lastName(self):
        return self.name.split()[-1]
    def giveRaise(self,percent):
        self.pay = int(self.pay * (1 + percent))
    def __str__(self):
        return '[%s: %s,%s]' % (self.__class__.__name__,self.name,self.pay)

class Manager(Person):
    def __init__(self,name,pay):
        Person.__init__(self,name,'mgr',pay)
    def giveRaise(self,percent,bonus=0.1):
        Person.giveRaise(self,percent + bonus)
        
if __name__ == '__main__':
    bob = Person('bob jason','emp',10000)
    sue = Person('sue lucy','std',0)
    tom = Manager('tom jones',20000)
    print(tom)
    db = shelve.open('persondb')
    for i in (bob,sue,tom):
        db[i.name] = i
    db.close
    db = shelve.open('persondb')
    print(db['bob jason'])

try:
    a = 1 / 0
    print('continue')
except ZeroDivisionError:
    print('error')
finally:
    print('finally')
print('over')

