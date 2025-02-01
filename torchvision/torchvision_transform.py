import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(
    './dataset', train = True, transform=dataset_trans, download= True
)
test_set = torchvision.datasets.CIFAR10(
    './dataset', train = False, transform=dataset_trans, download = True
)

img_0, target_0 = test_set[0]
print(img_0, 'with target number of ',target_0)

writer = SummaryWriter('logs')
for i in range(5):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()