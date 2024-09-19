from random import shuffle

import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataset_use.p10_dataset_transform import writer

data_trans = torchvision.transforms.ToTensor()
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=data_trans)

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f"epoch:{epoch}",imgs,step)
        step += 1
        # print(imgs)
        # print(targets)

writer.close()