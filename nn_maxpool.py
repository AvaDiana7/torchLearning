import torch
import torchvision
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float)

input = torch.reshape(input, (-1, 1, 5, 5))

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataLoader = DataLoader(dataset, batch_size=61, shuffle=True)


class jlNet(nn.Module):

    def __init__(self):
        super(jlNet, self).__init__()
        self.maxpool1 = nn.MaxPool2d(3, ceil_mode=False)

    def forward(self, x):
        output = self.maxpool1(x)
        return output


net = jlNet()
# output = net(input)
# print(output)

writer = SummaryWriter("./logs")
step = 0

for data in dataLoader:
    imgs, targets = data
    output = net(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)

    step += 1

writer.close()
