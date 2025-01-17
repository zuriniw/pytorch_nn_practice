import torch
from torch.utils.data import Dataset
from PIL import Image
import os

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
        return img, label

    def __len__(self):
        return len(self.imgs_path)

hol_root_dir = 'hymenoptera_data/train'
ant_label_dir, bee_label_dir = 'ants', 'bees'
ant_dataset = MyData(hol_root_dir, ant_label_dir)
bee_dataset = MyData(hol_root_dir, bee_label_dir)
train_dataset = ant_dataset + bee_dataset

print(len(train_dataset) == len(ant_dataset) + len(bee_dataset))    # True
# img_0, label_0 = ant_dataset[0]
# img_0.show()






