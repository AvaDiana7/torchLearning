import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor())

dataLoader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)


class jlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = jlNet()
print(net)


writer = SummaryWriter("logs")
step = 0
for data in dataLoader:
    imgs, targets = data
    output = net(imgs)
    # print(imgs.shape)
    #torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)
    # torch.Size([64, 6, 30, 30])
    # print(output.shape)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step += 1
