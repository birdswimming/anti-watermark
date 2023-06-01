import random
import torch
import os
import glob
from PIL import Image
from torchvision import transforms
from torchvision import utils

# mean and std channel values for places2 dataset
MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]


# reverses the earlier normalization applied to the image to prepare output
def unnormalize(x):
    x.transpose_(1, 3)
    x = x * torch.Tensor(STDDEV) + torch.Tensor(MEAN)
    x.transpose_(1, 3)
    return x


class My_Places (torch.utils.data.Dataset):

    def __init__(self, path_to_data):
        super().__init__()
        fh = open(path_to_data, 'r') #按照传入的路径和txt文本参数，打开这个文本，并读取内容
        img_paths = []
        mask_paths = []
        for line in fh:                #按行循环txt文本中的内容
            line = line.rstrip()       # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()   #通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            img_paths.append(words[0])
            mask_paths.append(words[1])
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.num_imgs  = len(self.img_paths)
        self.num_masks = len(self.mask_paths)
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STDDEV)])
        
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        gt_img = Image.open(self.img_paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))
        
        mask = Image.open(self.mask_paths[index])
        mask = self.mask_transform(mask.convert('RGB'))

        return gt_img * mask, mask, gt_img