from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2
# 通过 transforms.totensor 了解 transforms 的使用和 tensor 作为一种数据类型
# 第一步依然是 cmd+click 查看 transforms 的 source code
writer = SummaryWriter('logs')
img_path = 'hymenoptera_data/train/bees/16838648_415acd9e3f.jpg'
img = Image.open(img_path)
print(type(img))    # <class 'PIL.JpegImagePlugin.JpegImageFile'>

cv_img = cv2.imread(img_path)
print(type(cv_img))     # <class 'numpy.ndarray'>



############## ToTensor ############
trans_totensor = transforms.ToTensor()    # 这里制造了一个用于转化的机器
img_tensor = trans_totensor(img)
print(type(img_tensor))     # <class 'torch.Tensor'>


############## Normalize ############
writer.add_image('ToTensor', img_tensor)
print(img_tensor[0][0][0])

trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])     # 这里制造了一个用于算归一化的机器
img_norm = trans_norm(img_tensor)
# 根据source code，Normalization 的计算公式是：
#       output = (input - mean) / std
#       带入 0.5，即 output = (input - 0.5) *2 == 2 * input - 1, 可以把[0,1]domain --> [-1,1]domain

writer.add_image('Normalization', img_norm)
print(img_norm[0][0][0])



############## Resize ############
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)

writer.add_image('resize', img_resize)



############## Compose ############
trans_resize_2 = transforms.Resize(512)
#   PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
result = trans_compose(img)
print(type(result))
writer.add_image('compose',result,1)



############## RandomCrop ############
trans_randomcrop = transforms.RandomCrop((200,400))
trans_compose_croptensor = transforms.Compose([trans_randomcrop, trans_totensor])
for i in range (10):
    result_croptensor = trans_compose_croptensor(img)
    writer.add_image('croptensorHW', result_croptensor, i)
# randomcrop 可以用于做数据增广，加大数据集的量




writer.close()