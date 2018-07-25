def sort_quick(array):
    if len(array) < 2:
        return array
    else:
        base = array[0]
        less = [i for i in array if i < base]
        greater = [i for i in array if i > base]
        return sort_quick(less) + [base] + sort_quick(greater)

if __name__=='__main__':
    import numpy as np
    from time import time
    array = np.random.normal(0,1,100000)
    time_start = time()
    for i in range(3):
        array_sorted = sort_quick(array)
    time_end = time()
    print('排序耗时：',time_end - time_start)
    # print(np.round(array,3),'\n',np.round(array_sorted,3))