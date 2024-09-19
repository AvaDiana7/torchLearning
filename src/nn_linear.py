import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor())
dataLoader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)

class jlNet(nn.Module):

    def __init__(self):
        super(jlNet,self).__init__()
        self.linear1 = nn.Linear(3072,10)

    def forward(self,input):
        x = self.linear1(input)
        return x

net = jlNet()


for data in dataLoader:
    imgs, target = data
    imgs = torch.reshape(imgs,(64,1,1,-1))
    print(imgs.shape)
    output = net(imgs)
    print(output.shape)