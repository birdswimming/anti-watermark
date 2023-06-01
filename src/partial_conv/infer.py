import torch
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from my_places import My_Places
from my_places import unnormalize
import numpy as np
import cv2

test_data_path  = 'data/training_data/mix/train_partialConv.txt'
test_dataset =My_Places(path_to_data=test_data_path)

device = torch.device("cuda") # use GPU to train
model = torch.load('model/partitail_conv_net.pkl')
model.eval()
model.to(device)

test_loader  = DataLoader(dataset=test_dataset,  batch_size=4)
for i, (images_masked, masks, images) in enumerate(test_loader):

    image_masked  \
          = torch.tensor(list(i.numpy() for i in images_masked)).to(device)
    mask  = torch.tensor(list(m.numpy() for m in masks)).to(device)
    image = torch.tensor(list(t.numpy() for t in images))

    with torch.no_grad():
        output = model(image_masked, mask).cpu()
        mask = mask.cpu()
        output = (mask * image) + ((1 - mask)*output)
    break
    
# for image in output:
output = unnormalize(output)
output = output.numpy()
tmp = output[:,0,:,:].copy()
output[:,0,:,:] = output[:,2,:,:]
output[:,2,:,:] = tmp

output = np.transpose(output, (0, 2, 3, 1))

output = output * 255
output = output.astype(np.uint8)


cv2.imwrite('./tmp1.jpg', output[0])
cv2.imwrite('./tmp2.jpg', output[1])
cv2.imwrite('./tmp3.jpg', output[2])
cv2.imwrite('./tmp4.jpg', output[3])


    