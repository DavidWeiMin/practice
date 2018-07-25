import time
a = range(10000)
time_start = time.time()
def binary_search(item,list):
    high = len(list) - 1
    low = 0
    while high - low > 1:
        if item < list[(high + low) // 2]:
            high = (high + low) // 2
        else:
            low = (high + low) // 2
    if list[low] == item:
        return low
    else:
        return high

for i in a:
    b = binary_search(999,a)
print('元素所在位置:',b)
time_end = time.time()
print('算法用时：',time_end - time_start)



