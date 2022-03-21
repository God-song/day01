import torch
import numpy as np
#查看torch版本
#print(torch.__version__)

#创建张量
#利用numpy
# d=np.array(9)
# print(d,type(d),d.dtype)
# e=np.array([1,2,3])
# print(e,type(e))
# f=np.array([[1,2,3],[4,5,6]])
# _f=torch.tensor([[1,2,3],[4,5,6]])
# print(f,type(f),f.dtype)
# print(_f,type(_f),_f.dtype)
#快速创建
# a=np.zeros((5,4))
# _a=torch.zeros((5,4))
# print(a,a.dtype)
# print("==")
# print(_a,_a.dtype)
# print(a.shape,_a.shape)
# b=np.ones((3,4))
# print(b,b.dtype,b.shape)
# _b=torch.ones((3,4))
# print(_b)
# print(np.random.randn(1,2))
# print(torch.randn(2,2))


#修改张量
# a=np.arange(1,7).reshape(2,3)
# b=torch.arange(1,7).reshape(2,3)
# print(b)
# print(a,a.shape)
# print(a[1,2])
# a[1,2]=7
# print(a)
#张量切片
# print(a[:,2])
# a[:,2]=7
# print(a)
# a[:,:]=7
# print(a)
# print(b[1][:])
# print(b)


#张量的合并
# a=torch.arange(1,13).reshape(3,4)
# print(a,a.shape)
# b=torch.arange(1,5)
# print(b,b.shape)
# _b=b[None,:]
# # print(_b,_b.shape)
# # c=a+b
# # print(c)
# # print(c.shape)
# print(_b,_b.shape)
# c=torch.cat((a,_b),dim=0)
# print(c)

# a=np.arange(1,13).reshape(3,4)
# print(a,a.shape)
# b=np.arange(7,10)
# print(b,b.shape)
# _b=b[:,None]
# print(_b.shape)
# c=np.concatenate((a,_b),axis=1)
# print(c)

# a=torch.tensor([1,2])
# b=torch.tensor([3,4])
# print(a.shape)
# print(torch.stack([a,b],dim=1))