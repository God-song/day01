import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim

#定义x的值
# x=np.arange(0.1,10,0.1)
# #print(x)
# y=x**2
#print(y)
x=torch.arange(0.01,1,0.01)
y=3*x+4+torch.rand(99)
#print(x,y)
class Line(torch.nn.Module):
    #创建模型,构建模型
    def __init__(self):
        super(Line, self).__init__()
        #初始化模型的参数,Parameter是框架定义好的初始值，后面可以自动被优化
        self.w=torch.nn.Parameter(torch.rand(1))
        self.b=torch.nn.Parameter(torch.rand(1))

        #前向计算
    def forward(self,x):
        return self.w*x+self.b

if __name__=="__main__":
    #初始化对象
    line=Line()
    #创建优化器,可以优化数据
    opt=optim.SGD(line.parameters(),lr=0.1)

    #构建损失函数
    #直接使用框架里面的函数，得到的会是平均之后的损失值
    #loss_func=torch.nn.MSELoss()
    plt.ion()
    for j in range(20):
        for _x,_y in zip(x,y):
            z=line.forward(_x)
            loss=(z-_y)**2
            #将之前的梯度数值清空
            opt.zero_grad()
            loss.backward()
            #跟新梯度
            opt.step()
            #直接line.w得到的是一个tensor张量,要显示里面具体标量
            print(line.w.item(),line.b.item(),loss.item())
            plt.cla()
            plt.plot(x,y,".")
            #这个地方会报错，w,b为张量不能直接和
            v=[line.w.detach()*i+line.b.detach() for i in x]
            plt.plot(x,v)
            plt.pause(0.01)
    plt.ioff()

# y=1/(1+np.exp(-x))
# plt.plot(x,y)
# #显示图像
# plt.show()
