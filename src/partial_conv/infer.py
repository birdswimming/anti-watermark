import torch
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from my_places import My_Places
from my_places import unnormalize
test_data_path  = 'data/testing_data/cropped/test.txt'
test_dataset =My_Places(path_to_data=test_data_path)

device = torch.device("cuda") # use GPU to train
model = torch.load('model/partitail_conv_net.pkl')
model.eval()
model.to(device)

test_loader  = DataLoader(dataset=test_dataset,  batch_size=test_dataset.__len__())
for i, (image_masked, mask, image) in enumerate(test_loader):
    with torch.no_grad():
        output = model(image_masked, mask)
        output = (mask * image) + ((1 - mask)*output)
    
for image in output:
    image = unnormalize(image)
    
    