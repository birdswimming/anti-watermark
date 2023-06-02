import torch
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from my_detection import MyDetection
from my_detection import get_transforms
test_data_path  = 'data/testing_data/data.txt'
test_dataset =MyDetection(datatxt=test_data_path,  transform=get_transforms(False))

device = torch.device("cuda") # use GPU to train
model = torch.load('model/fast_rcnn.pkl')
model.eval()
model.to(device)

test_loader  = DataLoader(dataset=test_dataset,  batch_size=1)

boxes = []
labels = []
scores = []

for i, (images, target) in enumerate(test_loader):
    image_batched = list(img.to(device) for img in images)
    pred = model(image_batched)
    boxes.append(pred[0]['boxes'].cpu().detach().numpy())
    labels.append(pred[0]['labels'].cpu().detach().numpy())
    scores.append(pred[0]['scores'].cpu().detach().numpy())

import pprint
pprint.pprint(boxes)
pprint.pprint(labels)
pprint.pprint(scores)
print(len(boxes))
print(len(labels))
print(len(scores))

f = open("./data/testing_data/bbox_result.txt", "w")
for i,result in enumerate(boxes):
    find = 0
    for j,score in enumerate(scores[i]):
        if score > 0.7:
            box = boxes[i][j]
            x_min = box[0]/600
            y_min = box[1]/600
            x_max = box[2]/600
            y_max = box[3]/600
            f.write(f"{x_min} {y_min} {x_max} {y_max}\n")
            find = 1
            break
    if find == 0:
        f.write(f"0 0 0 0\n")