import torch
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from my_detection import MyDetection
from my_detection import get_transforms
test_data_path  = 'data/testing_data/cropped/test.txt'
test_dataset =MyDetection(datatxt=test_data_path,  transform=get_transforms(False))

device = torch.device("cuda") # use GPU to train
model = torch.load('model/faster_rcnn.pkl')
model.eval()
model.to(device)

test_loader  = DataLoader(dataset=test_dataset,  batch_size=test_dataset.__len__())
for i, (images, target) in enumerate(test_loader):
    image_batched = list(img.to(device) for img in images)
    pred = model(image_batched)
    
print(type(pred))
print(type(pred[0]))
f = open("./data/testing_data/cropped/bbox_result.txt", "w")
for result in pred:
    boxes = result['boxes'][result['scores']>0.8].cpu().squeeze().tolist()
    
    print(boxes)
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0
    if len(boxes)!=0:
        x_min = boxes[0]/600
        y_min = boxes[1]/600
        x_max = boxes[2]/600
        y_max = boxes[3]/600
    f.write(f"{x_min} {y_min} {x_max} {y_max}\n")