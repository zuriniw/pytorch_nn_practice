import torchvision.datasets

# 准备的测试数据集
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10('./dataset', False, transform = torchvision.transform.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img, target)
