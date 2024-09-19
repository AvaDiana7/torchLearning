from torch import nn
import torch

class jlNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward(self,input):
        input += 1
        return input

net = jlNet()
x = torch.tensor(2)
res = net(x)
print(res)