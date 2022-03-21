import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os


file_dir=os.path.abspath(__file__)
base_dir=os.path.dirname(file_dir)
print(file_dir)
print(base_dir)

im=Image.open(os.path.join(base_dir,"1.jpg"))
print(im.shape)
#im.show()]
# im=np.array(im)
# im=im.transpose([1,0,2])#torch.permute
# im=Image.fromarray(im)
# im.show()
#im.reshape(



