import torch
# from torch.utils.data._utils
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import cv2

# 这里以SummaryWriter作为abstract class, 创建了instance writer；
# 根据cmd+click的文档，第一个arg是 Save directory location
writer = SummaryWriter('logs')


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir

        self.path = os.path.join(self.root_dir, self.label_dir)
        self.imgs_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.imgs_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label, img_item_path

    def __len__(self):
        return len(self.imgs_path)

hol_root_dir = 'hymenoptera_data/train'
ant_label_dir, bee_label_dir = 'ants', 'bees'
ant_dataset = MyData(hol_root_dir, ant_label_dir)
bee_dataset = MyData(hol_root_dir, bee_label_dir)

ant_imgs_path = [os.path.join(ant_dataset.root_dir, ant_dataset.label_dir, img_path) for img_path in ant_dataset.imgs_path]
print(ant_imgs_path)


# 因为image_add 方法要求图片格式是numpy，所以需要把图片格式为np array
ant_imgs_array = []
for ant_img_path in ant_imgs_path:
    ant_img_jpg = Image.open(ant_img_path)
    ant_img_array = np.array(ant_img_jpg)
    ant_imgs_array.append(ant_img_array)
print(ant_imgs_array[0].shape)      # (375, 500, 3)
# 这里发现图片格式为 WHC, 即宽高三通道
# 但是 .add_image 的要求是 CWH，如果不是，需要传入对应的 arg ：data format

# 每一个图片对应一个step i
i = 0
for ant_img_array in ant_imgs_array[:10]:
    i += 1
    writer.add_image('train', ant_img_array, i, dataformats='HWC')

# 绘制一个diagram
for i in range(100):
    writer.add_scalar('y = 2x', 2*i, i)    # 这里的参数分别为：title，因变量，自变量


writer.close()



