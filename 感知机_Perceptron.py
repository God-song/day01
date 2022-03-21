import torch
from torch import optim
from matplotlib import  pyplot as plt

#与 问题
x=torch.tensor([[1.,1],[1,0],[0,1],[0,0]])
y=torch.tensor([[1.],[0],[0],[0]])
test_X=torch.tensor([[2,1],[0,0.5],[0.8,0],[0.8,0.9],[0.6,0.9]])
class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.w1=torch.nn.Parameter(torch.randn(2,4))
        self.b=torch.nn.Parameter(torch.randn(4))
        self.w2=torch.nn.Parameter(torch.randn(4,1))
        self.b2=torch.nn.Parameter(torch.randn(1))


    def forward(self,x):

        fc1=torch.matmul(x,self.w1)+self.b
        fc2=torch.matmul(fc1,self.w2)+self.b2

        return fc2


if __name__=="__main__":

    line=net()
    opt = optim.SGD(line.parameters(), lr=0.01)

    # 构建损失函数
    # 直接使用框架里面的函数，得到的会是平均之后的损失值
    # loss_func=torch.nn.MSELoss()
    #plt.ion()
    loss_func = torch.nn.MSELoss()
    for j in range(700):

        z = line.forward(x)
        loss =loss_func(z,y)
        # 将之前的梯度数值清空
        opt.zero_grad()
        loss.backward()
        # 跟新梯度
        opt.step()
        # 直接line.w得到的是一个tensor张量,要显示里面具体标量
        print(loss.item())
        # plt.cla()
        # plt.plot(x, y, ".")
        # 这个地方会报错，w,b为张量不能直接和
        # v = [line.w.detach() * i + line.b.detach() for i in x]
        # plt.plot(x, v)
        # plt.pause(0.01)
    #plt.ioff()
    print("============")

    print(line.forward(test_X)>0.5)


