import torch
import torchvision
from torch import nn
from torch.nn import Flatten
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10("../dataset",train=False,transform=transform)

dataloader = DataLoader(dataset=train_set,batch_size=64,drop_last=True)

class jlNet(nn.Module):
    def __init__(self):
        super(jlNet, self).__init__()
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
        x = self.model(x)
        return x


net = jlNet()
for epoch in range(5):
    for data in dataloader:
        imgs, targets = data
        output = net(imgs)
        loss = CrossEntropyLoss()
        res_loss = loss(output,targets)
        res_loss.backward()
        print(res_loss)
