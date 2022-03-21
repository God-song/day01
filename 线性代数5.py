import random

import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
a=torch.arange(-20.0,20.0)/20
#print(a)
xs=torch.unsqueeze(a,dim=1)
print(xs)
ys=[i.pow(3)*random.randint(1,6) for i in xs]
print(ys)
#将ys列表类型，转换乘tensor类型
ys=torch.stack(ys)
# plt.plot(xs,ys,".")
#
# plt.show()
print(ys)
#采用刚才的模型解决，看效果如何
class Line(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1=nn.Linear(1,20)
        self.s1=nn.Sigmoid()
        self.layer2 = nn.Linear(20, 64)
        self.s2 = nn.Sigmoid()
        self.layer3 = nn.Linear(64, 128)
        self.s3 = nn.Sigmoid()
        self.layer4 = nn.Linear(128, 64)
        self.s4 = nn.Sigmoid()
        self.layer5 = nn.Linear(64, 1)



    def forward(self,x):
        fc1=self.layer1(x)
        fc1=self.s1(fc1)
        fc2 = self.layer2(fc1)
        fc2 = self.s2(fc2)
        fc3 = self.layer3(fc2)
        fc3 = self.s3(fc3)
        fc4 = self.layer4(fc3)
        fc4 = self.s4(fc4)
        fc5 = self.layer5(fc4)
        return fc5

if __name__=="__main__":
    #创建神经网络
    net=Line()
    #开始构建优化模型
    opt=optim.Adam(net.parameters())
    #创建网络损失函数
    loss_func=nn.MSELoss()
    #循环训练数据
    plt.ion()
    for i in range(1000):
        #开始正向获得数据
        out=net.forward(xs)
        loss=loss_func(out,ys)
        #清空梯度
        opt.zero_grad()

        #求导数
        loss.backward()
        #更新梯度
        opt.step()

        print(loss)
        plt.cla()
        plt.plot(xs,ys,".")

        plt.plot(xs,out.detach())

        plt.pause(0.001)

    plt.ioff()
    plt.show()

