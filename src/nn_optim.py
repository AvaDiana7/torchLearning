import torch
import torchvision
from torch import nn
from torch.nn import Flatten
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10("../dataset", train=False, transform=transform)

dataloader = DataLoader(dataset=train_set, batch_size=128, drop_last=True)


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
loss = CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
for epoch in range(20):
    print(f"epoch:{epoch}")
    running_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        imgs, targets = data
        output = net(imgs)
        res_loss = loss(output, targets)
        res_loss.backward()
        optimizer.step()
        running_loss += res_loss

    print(running_loss)
