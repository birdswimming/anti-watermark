import torch
import os
import cv2
from torchvision import datasets, models
import albumentations as A  # our data augmentation library
from albumentations.pytorch import ToTensorV2

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p = 0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            ToTensorV2()
        ])
    return transform


class MyDetection(datasets.VisionDataset):
    def __init__(self, datatxt, transform=None, target_transform=None): #初始化一些需要传入的参数
        fh = open(datatxt, 'r') #按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []                      #创建一个名为img的空列表，一会儿用来装东西
        for line in fh:                #按行循环txt文本中的内容
            line = line.rstrip()       # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()   #通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append([words[0],[float(words[1]), float(words[2]), float(words[3]), float(words[4])]]) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
                                        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_image(self, path):
        image = cv2.imread(os.path.join(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def __getitem__(self, index):
        image = self._load_image(self.imgs[index][0])
        
        if self.transform is not None:
            image = self.transform(image = image)['image']
        
        new_boxes = [] # convert from xywh to xyxy
        #x_min, y_min, x_max, ymax
        x_min = int(self.imgs[index][1][0] * 600)
        y_min = int(self.imgs[index][1][1] * 600)
        x_max = int(self.imgs[index][1][2] * 600)
        y_max = int(self.imgs[index][1][3] * 600)
        new_boxes.append([x_min, y_min, x_max, y_max])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {} # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([1], dtype=torch.int64)
        targ['image_id'] = torch.tensor([index])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
        
        return image.div(255), targ # scale images
    def __len__(self):
        return len(self.imgs)