import random

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as  plt
from torch import optim

xs=torch.arange(0.01,1,0.1)
#将xs 升维
print(xs)

xs=torch.unsqueeze(xs,dim=1)
print(xs)
# print(xs)
# print(xs.shape)
# print(torch.rand(99))
# print(torch.rand(99).shape)
ys=3*xs+4+random.randint(1,10)
#print(xs)
# print(ys)
# print(ys.shape)
# plt.plot(xs,ys,".")
# plt.show()
# w1=torch.nn.parameter(torch.randn(1,20))
# print(w1)
class Line(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.w1=torch.nn.Parameter(torch.randn(1,20))
        # self.b1=torch.nn.Parameter(torch.randn(20))
        # self.w2 = torch.nn.Parameter(torch.randn(20, 64))
        # self.b2 = torch.nn.Parameter(torch.randn(64))
        # self.w3 = torch.nn.Parameter(torch.randn(64, 128))
        # self.b3 = torch.nn.Parameter(torch.randn(128))
        # self.w4 = torch.nn.Parameter(torch.randn(128, 64))
        # self.b4 = torch.nn.Parameter(torch.randn(64))
        # self.w5 = torch.nn.Parameter(torch.randn(64, 1))
        # self.b5 = torch.nn.Parameter(torch.randn(1))

        #简化，如何创建神经网络,定义网络层的方式
        # self.layer1=nn.Linear(1,20)
        # self.layer2 = nn.Linear(20, 64)
        # self.layer3 = nn.Linear(64, 128)
        # self.layer4 = nn.Linear(128, 64)
        # self.layer5 = nn.Linear(64, 1)
        #再简化，定义网络块
        self.fc_layer=nn.Sequential(
            nn.Linear(1,20),
            nn.Linear(20, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        )


    def forward(self,x):
        #叉乘
        # fc1=torch.matmul(x,self.w1)+self.b1
        # fc2=torch.matmul(fc1,self.w2)+self.b2
        # fc3 = torch.matmul(fc2, self.w3) + self.b3
        # fc4 = torch.matmul(fc3, self.w4) + self.b4
        # fc5 = torch.matmul(fc4, self.w5) + self.b5

        #简化
        # fc1=self.layer1(x)
        # fc2 = self.layer2(fc1)
        # fc3 = self.layer3(fc2)
        # fc4 = self.layer4(fc3)
        # fc5 = self.layer5(fc4)
        return self.fc_layer(x)

if __name__=="__main__":
    net=Line()

    opt=optim.Adam(net.parameters())

    loss_func=torch.nn.MSELoss()
    plt.ion()
    for i in range(200):
        out=net.forward(xs)
        loss=loss_func(out,ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i%5==0:
            #显示一下
            print(loss.item())

            plt.cla()

            plt.plot(xs,ys,".")

            plt.plot(xs,out.detach())

            plt.pause(0.001)
            plt.title(loss)
            #plt.show()
    plt.ioff()
    plt.show()