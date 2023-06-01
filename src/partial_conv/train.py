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

# train_data_path = 'data/training_data/mix/overfit.txt'
train_data_path = 'data/training_data/mix/train_partialConv.txt'
train_dataset=My_Places(path_to_data=train_data_path)
# model = PartialConvUNet()
model = torch.load('model/partitail_conv_net.pkl')
# model = torch.load('model/overfit.pkl')
model.train()

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size = 128, shuffle=True, num_workers=4, collate_fn=collate_fn)
device = torch.device("cuda") # use GPU to train

def requires_grad(param):
	return param.requires_grad

lr = 0.001
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=lr)
loss_func = CalculateLoss().to(device)

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    for images, masks, targets in tqdm(loader):
                
        image  = torch.tensor(np.array([i.numpy() for i in images ])).to(device)
        mask   = torch.tensor(np.array([m.numpy() for m in masks  ])).to(device)
        target = torch.tensor(np.array([t.numpy() for t in targets])).to(device)
        
        output = model(image, mask)
        
        loss = 0.0
        loss_dict = loss_func(image, mask, output, target)
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
# torch.save(model, 'model/overfit.pkl')