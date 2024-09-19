import torch
import torchvision
from torch import nn
from torch.nn import Flatten

transform = torchvision.transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10("../dataset", train=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("../dataset", train=False, transform=transform)


class jlNet(nn.Module):
    def __init__(self):
        super(jlNet, self).__init__()
        # self.conv1 = nn.Conv2d(3,32,5,1,2)
        # self.maxpool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(32,32,5,1,2)
        # self.maxpool2 = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(32,64,5,1,2)
        # self.maxpool3 = nn.MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = nn.Linear(1024,64)
        # self.linear2 = nn.Linear(64,10)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model(x)
        return x


net = jlNet()
print(net)
data = torch.ones((64, 3, 32, 32))

output = net(data)
print(output.shape)
