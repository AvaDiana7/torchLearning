import torchvision
from torchvision import transforms
import PIL
from torch.utils.tensorboard import SummaryWriter

dataset_trans = transforms.Compose([
    transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_trans,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_trans,download=True)

# print(test_set[0])
# print(train_set[0])
# print(train_set.classes)
# img, target = test_set[0]
#
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()