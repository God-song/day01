#张量运算

import numpy as np
import torch

# a=np.array([1,2,3])
# print(a*2)
# print(a**2)
# print(np.exp(a))
# b=np.array([4,5,6])
# print(a*b)
# print(a.dot(b))
#广播
# a=np.array([[1,2,3],[4,5,6]])
# b=np.array([4,5,6])
# print(a+b)
# a=np.arange(1,13).reshape(3,4)
# b=np.arange(1,7).reshape(3,2)
# #print(a+b)
# #_b=b.stack([b,b],dim=1)
# _b=np.tile(b,(1,2))  #torch.repeat
# print(_b,_b.shape)
# print(a+_b)



#矩阵点乘
#mxn nx k =m k
# a=np.arange(1,7).reshape(2,3)
# b=np.arange(1,7).reshape(3,2)
# print(np.dot(a,b))
# a=torch.arange(1,7).reshape(2,3)
# b=torch.arange(1,7).reshape(3,2)
# print(a@b,torch.matmul(a,b))
#张量点乘

# a=torch.arange(1,13).reshape(2,2,3)
# print(a)
# b=torch.arange(1,13).reshape(2,3,2)
# print(b)
# print(a@b)

#逆
# a=torch.arange(1.0,5).reshape(2,2)
# print(torch.inverse(a))

#统计函数

# a=torch.arange(1,7).reshape(2,3)
# print(a[1:,:])
# print(a[1:,:].sum())
# print(a.sum(dim=1))
# print(a.sum(dim=0))
#@a=torch.arange(1,13).reshape(2,2,3)
# print(a.shape)
# print(a)
# print(a.sum(dim=0))
# print(a.sum(dim=1))
# print(a.sum(dim=2))
#a=np.arange(1,13).reshape(2,2,3)
# print(a.sum(1))
# #平均值
# print(a.mean(1))
# #标准差
# print(a.std(1))
#n(0,1)平均值为0 ，标准差为1,标准正太分布
#讲述据转换乘正太分布
#a-a.mean()/a.std()

# print(a)
# print(a.max(2))
# print(a.argmax(2))

#张量合并
a=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(a.shape)
#形状变形
print(a.permute(0,2,1))
print(a.shape)




