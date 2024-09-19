import torch
import torchvision

model = torch.load("vgg16_method1.pth")

#方法2
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load()
print(vgg16)