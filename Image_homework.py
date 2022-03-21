from PIL import Image
import torch
import os
import numpy as np
file_name=os.path.abspath(__file__)
file_base=os.path.dirname(file_name)

print(file_name)
print(file_base)
img_path=os.path.join(file_base,"1.jpg")
print(img_path)

img1=Image.open(img_path)
# #img1.show()
# img_array=torch.tensor(img1)
img_array=np.array(img1)
print(img_array.shape)
img_torch=torch.from_numpy(img_array)
print(img_torch.shape)
#将图片切成四个
img_torch=img_torch.reshape(2,800,2,2560//2,3)
print(img_torch.shape)

img_torch=img_torch.permute(0,2,1,3,4)
print(img_torch.shape)
img_torch=img_torch.reshape(4,800,1280,3)
print(img_torch.shape)
def look(a):
    for i in a:
        img_a = np.array(i)
        b = Image.fromarray(img_a)
        b.show()
# for i in img_torch:
#     img_a=np.array(i)
#     a=Image.fromarray(img_a)
#     a.show()

#接下来将imgtorch 转变成一个图片
img_torch=img_torch.reshape(2,2,800,1280,3)
#look(img_torch)
img_torch=img_torch.permute(0,2,1,3,4)
print(img_torch.shape)
img_torch=img_torch.reshape(1600,2560,3)
print(img_torch.shape)
a=np.array(img_torch)
img_torch=Image.fromarray(a)
#print(img_torch.shape)
img_torch.show()


