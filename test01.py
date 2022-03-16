import numpy as np
import torch
import matplotlib.pyplot as plt

#定义x的值
# x=np.arange(0.1,10,0.1)
# #print(x)
# y=x**2
#print(y)
x=np.arange(-9.9,10,0.1)
y=1/(1+np.exp(-x))
plt.plot(x,y)
#显示图像
plt.show()
