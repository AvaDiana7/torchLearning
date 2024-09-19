import torch
import torchvision
from torch import nn

from torch.utils.data import DataLoader

vgg16 = torchvision.models.vgg16()

# 方法1
torch.save(vgg16,"vgg16_method1.pth")

# 方法2
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

