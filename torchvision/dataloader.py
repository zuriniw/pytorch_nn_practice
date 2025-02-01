import torchvision
from torch.utils.data import DataLoader

# 准备的测试数据集
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10('./dataset', False, transform = torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=0, drop_last=False)

# 测试数据集中第一张样本
img, target = test_data[0]
print(len(img), len(img[0]), target)


# board
# 这里的一个for循环就相当于从池里面抓完一轮牌

writer = SummaryWriter('/Users/ziru/Documents/GitHub/pytorch_nn_practice/torchvision/logs/dataloader')        # 这里的dataloader是自定义的log保存的路径名字
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f'Epoch{epoch}', imgs, step)      # 注意这里的images的s：这说明是批量加图
        step += 1

writer.close()
