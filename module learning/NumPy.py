import numpy as np
# # ndarray的基本使用
# #######################################################################

# # 将list转为ndarray
# a1 = [1,2,3,4];                                             print(a1)
# array1 = np.array(a1);                                      print(array1)

# # 嵌套序列生成ndarray
# a2 = [[1,2,3],[2,4,5]];                                     print(a2)
# array2 = np.array(a2);                                      print(array2)               # 指定数组元素类型

# # 获得创建的ndarray的信息
# print(array1.shape);                                        print(array2.shape)
# print(array1.ndim);                                         print(array2.ndim)
# print(array1.dtype);                                        print(array2.dtype)
# print(type(array1.shape));                                  print(type(array1.ndim))
# print(type(array1.dtype))

# # 创建特殊的ndarray
# zero1 = np.zeros(5);                                        print(zero1)
# zero2 = np.zeros((2,4));                                    print(zero2)
# zero3 = np.zeros_like(array2);                              print(zero3)               # 不仅维数相同，数据类型也相同
# one1 = np.ones(4,dtype=np.int8);                            print(one1)
# one2 = np.ones((4,2));                                      print(one2)
# one3 = np.ones_like(array2);                                print(one3)
# eye1 = np.eye(3);                                           print(eye1)                # 可以创建矩形矩阵
# eye2 = np.eye(2,3,0);                                       print(eye2)
# identity1 = np.identity(4);                                 print(identity1)           # 只能创建方形矩阵
# empty1 = np.empty(4);                                       print(empty1)
# empty2 = np.empty((2,3));                                   print(empty2)
# empty3 = np.empty_like(array2);                             print(empty3)
# array3 = np.arange(15);                                     print(array3)              # array4 = np.arange((4,3)) # 只能输入标量

# # 修改ndrray元素类型
# one2.astype(one1.dtype);                                    print(one2)                # one2并未改变
# one2 = one2.astype(one1.dtype);                             print(one2)
# one2 = one2.astype(np.float16);                             print(one2)
# ######################################################################

# # 运算广播
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# array4 = np.array([[1,2,3],[4,5,6]]);                       print(array4 * array4)
# print(array4 - array4);                                     print(1 / array4)
# print(array4 ** 0.5);                                       print(array4 - 1)
# array4[0] = 0;                                              print(array4)
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # 索引与切片
# # ********************************************************************
# array5 = np.array([0,1,2,3,4,5,6,7,8,9]);                   print(array5)
# print(array5[5]);                                           print(array5[0:10:1])
# print(array5[::1]);                                         print(array5[::-1])
# array6 = array5[0:10:2];                                    print(array6)
# array6[2] = 111;                                            print(array5)                  # 切片得到原矩阵的视图
# array7 = np.array([[1,2,3],[4,5,6],[7,8,9]]);               print(array7[2][2])            # 两种索引方式
# print(array7[2,2]);                                         print(array7[0:2])
# array8 = [[1,2,3],[4,5,6],[7,8,9]];                         print(array8[2][2])            # 只能以这种方式索引，不能[2,2]
# array8[0] = 0;                                              print(array8)
# array9 = array8[2];
# array9 = [4,4,4];                                           print(array8)
# # ************************************************************************

# # 布尔型索引
# ###########################################################################
# names = np.array(['Bob','Joe','Will','Bob','Will','Joe']);  print(names)
# data = np.random.randn(6,4);                                print(data)
# print(names == 'Bob');                                      print(data[names == 'Bob',2:])
# print(data[~(names == 'Bob')])
# data[data < 0] = 0;                                         print(data)
# ###########################################################################

# # 花式索引
# array10 = np.random.randn(4,2);                             print(array10)
# print(array10[[3,1,2,0]])

# 线性代数
# ----------------------------------------------------------------------------
# # 点乘
# x = np.array([[1,2,3],[2,2,1],[4,2,2]])
# y = np.ones((3,1))
# z = np.dot(x,y)
# print(z)
# # 转置
# x_t = x.T
# print(x_t)
# # 逆
# x_inv = np.linalg.inv(x)
# print(x_inv)
# print(x_inv.dot(x))
# print(x * x_inv)
# # 行列式
# det = np.linalg.det(x)
# print(det)
# # 
# eig = np.linalg.eig(x)
# print(eig)
# eigvals = np.linalg.eigvals
# print(eigvals)
# 随机数矩阵生成
# samples1 = np.random.normal(loc=0,scale=1,size=(4,4)) # 随机生成4×4的矩阵，均值为0，标准差为1
# print(samples1)
# samples2 = np.random.randint(low=1,high=20)
# print(samples2)
# samples3 = np.random.binomial(10,0.5,size=(2,2))
# print(samples3)
# 存取矩阵文件
# ---------------------------------------------------------------------------
a=np.array([3,1,4])
b=np.array([2,3,1])
c=np.where(a<b,a,b)
print(c)