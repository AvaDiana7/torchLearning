import torchvision
import torch
from torch import nn

# dataset = torchvision.datasets.ImageNet(root="../dataset",split="train",download=True
#                                         ,transform=torchvision.transforms.ToTensor())

vgg16_true = torchvision.models.vgg16(weights="DEFAULT")
vgg16_false = torchvision.models.vgg16()

train_set = torchvision.datasets.CIFAR10("../dataset",train=True,transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))

vgg16_false.classifier[6] = nn.Linear(4096,10)
