import torch
import numpy as np
#最小二乘法
#行列式,衡量矩阵大小，行列相等==方阵

# a=np.arange(1,5).reshape(2,2)
# print(a)
# print(np.linalg.det(a))
#a.t *x 负一次方 x.h y
# x=np.matrix(np.array([[1],[3],[7]]))
# y=np.matrix(np.array([[1],[3],[7]]))*4
# print(x)
# print(y)
# print("==========")
# print((x.T@x).I@x.T@y)

#求逆有很大的局限性.求逆很复杂，随着矩阵变大，计算量指数增长,解决维度较低，参数较少，传统机器学习常用

# #单位矩阵
# a=np.eye(4)
# print(a)
# #对角矩阵
# b=np.diag([1,2,3,4])
# print(b)
# #下三角矩阵
# a=np.tri(3,3)
# print(a)

#零矩阵
# a=np.zeros((3,3))
# print(a)
# b=torch.zeros((3,3))
# print(b)
#
# #1矩阵
# a=np.ones((3,3))
# print(a)

#