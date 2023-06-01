import torch
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torchvision import transforms
from my_dataset import MyDataset
#设置随机种子以便复现
torch.manual_seed(1)
EPOCH=50
BATCH_SIZE=4
LR=0.00001

 
#根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
train_data_path = 'data/training_data/cropped/test.txt'
test_data_path  = 'data/training_data/cropped/test.txt'
train_data=MyDataset(datatxt=train_data_path, transform=transforms.ToTensor())
test_data =MyDataset(datatxt=test_data_path,  transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = Data.DataLoader(dataset=test_data,  batch_size=test_data.__len__())


Net = resnet18(num_classes=4)
# 若存在cuda环境，即可使用注释代码
Net = Net.cuda()
# print(Net)
# 优化器
optimizer = torch.optim.Adam(Net.parameters(),lr=LR)
# 损失函数，分类问题
loss_fn = nn.MSELoss()

for epoch in range(EPOCH):
    # 迭代运行
    for step, (x, y) in enumerate(train_loader):
        # print(x.size())
        # print(y.size())
        x = x.cuda()
        y = y.cuda()
        output = Net(x)
        # print(output)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('Epoch: ',epoch, '| train loss: %.4f' % loss.data)
        if(step % 10 == 0):
            for test_step, (test_x, test_y) in enumerate(test_loader):
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                test_output = Net(test_x)
                loss = loss_fn(test_output, test_y)
                print('Epoch: ',epoch, '| loss: %.4f' % loss.data)
            
            # # 可以单独进行模型的测试
            # test_output = Net(test_x)
            # # 1代表索引列，因为刚好匹配到0-9，获取概率高的
            # # pre_y = torch.max(test_output, 1)[1].data.squeeze()
            
            # pre_y = torch.max(test_output, 1)[1]
            # # 获取准确率
            # accurary = float((pre_y == test_y).sum()) / float(test_y.size(0))
            # print('Epoch: ',epoch, '| train loss: %.4f' % loss.data, '| test accurary: %.2f' % accurary)
torch.save(Net, 'model/locate_overfit.pkl')

with open("./data/training_data/cropped/bbox_result.txt", "w") as f:
    for l in test_output:
        f.write(f"{l[0]} {l[1]} {l[2]} {l[3]}\n")
    

    
