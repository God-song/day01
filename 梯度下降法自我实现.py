import numpy
import numpy as np
from matplotlib import pyplot as plt

x=numpy.arange(0.1,1,0.01)
print(x)
y=[3*i+4+numpy.random.random() for i in x]
print(y)
#plt.plot(x,y,".")
#plt.show()
w = numpy.random.random()
b = numpy.random.random()
plt.ion()
for j in range(100):
    for _x,_y in zip(x,y):
        #定义模型参数

        h=w*_x+b

        o=h-_y
        loss=o**2

        #求导数
        dw=-2*o*_x
        db=-2*o
        print(dw,db,loss)
        #后向学习

        w=dw*0.1+w
        b=db*0.1+b
        plt.cla()
        plt.plot(x,y,".")
        v=[w*i+b for i in x]
        plt.plot(x,v)
        plt.show()
        plt.pause(0.01)
    plt.ioff()