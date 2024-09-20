import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from Dataset_use.p10_dataset_transform import writer
from model import *
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集
train_set = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10("../dataset", train=False, transform=ToTensor())

train_dataloader = DataLoader(train_set, batch_size=64)
test_dataloader = DataLoader(test_set, batch_size=64)

train_set_len = len(train_set)
test_set_len = len(test_set)
print(f"训练集的大小为：{train_set_len}")
print(f"测试集的大小为：{test_set_len}")

net = jlNet()
net = net.to(device)

# loss
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0

# 训练轮数
epoch = 20

# writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print(f"-------第 {i + 1} 轮训练开始--------")

    net.train()
    # 训练开始
    for data in train_dataloader:
        optimizer.zero_grad()
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = net(imgs)
        # 优化器优化
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0 or total_train_step == 0:
            end_time = time.time()
            print(end_time-start_time)
            print(f"训练次数: {total_train_step},loss: {loss.item()}....")
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试开始
    net.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss
            accuracy = (output.argmax(1) == targets).sum().item()
            total_accuracy += accuracy

    print("整体测试集loss为:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_set_len))
    total_test_step += 1
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # torch.save(net, "jlNet_{}.pth".format(total_test_step))
    print("模型保存")

# writer.close()
