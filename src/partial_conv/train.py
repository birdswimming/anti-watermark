import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import math
import sys
from tqdm import tqdm # progress bar
from my_places import My_Places
from loss import CalculateLoss
from partial_conv_net import PartialConvUNet

train_data_path = 'data/training_data/cropped/train.txt'
train_dataset=My_Places(path_to_data=train_data_path)
model = PartialConvUNet()

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
device = torch.device("cuda") # use GPU to train

def requires_grad(param):
	return param.requires_grad

lr = 0.0001
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=lr)
epoch = 10
loss_func = CalculateLoss().to(device)

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    for image, mask, target in tqdm(loader):
        output = model(image, target)
        loss_dict = loss_func(image, mask, output, target)
        loss = 0.0
        for key, value in loss_dict.items():
            loss += value
        # Resets gradient accumulator in optimizer
        optimizer.zero_grad()
        # back-propogates gradients through model weights
        loss.backward()
        # updates the weights
        optimizer.step()
    print("Epoch {}, Loss {}".format(epoch, loss))

num_epochs=10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
torch.save(model, 'model/partitail_conv_net.pkl')