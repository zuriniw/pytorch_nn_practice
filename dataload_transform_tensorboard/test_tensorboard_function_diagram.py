import torch
from torch.utils.tensorboard import SummaryWriter

# 这里以SummaryWriter作为abstract class, 创建了instance writer；
# 根据cmd+click的文档，第一个arg是 Save directory location
writer = SummaryWriter('logs')

for i in range(100):
    writer.add_scalar('y = 2x', 2*i, i)    # 这里的参数分别为：title，因变量，自变量


