import torch
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from my_dataset import MyDataset

def locate_water_mark(test_loader):
    net = torch.load('model/locate.pkl')
    net.eval()
    net = net.cuda()
    result = None
    for step, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        out = net(x)
        if step == 0:
            result = out
        else:
            result = torch.cat((result, out), 0)
    result = result.cpu()
    return result
test_data_set = MyDataset(datatxt='data/training_data/cropped/test.txt', transform=transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_data_set, batch_size=50)
label = locate_water_mark(test_loader)
print (label)