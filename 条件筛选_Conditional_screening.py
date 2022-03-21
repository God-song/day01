import torch
# a=torch.arange(1,7)
# print(a>3)
# print(a[a>3])
# print(torch.nonzero(a>3))
#扩维
a=torch.arange(1,7).reshape(2,3)
print(a)
print(a[a>2])
print(torch.nonzero(a>2))