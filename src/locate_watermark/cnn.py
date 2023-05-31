import torch
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
#设置随机种子以便复现
torch.manual_seed(1)
EPOCH=1
BATCH_SIZE=4
LR=0.001
from PIL import Image
import torch
 
class MyDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
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
 
    def __getitem__(self, index):    #这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB') #按照path读入图片from PIL import Image # 按照路径读取图片
 
        if self.transform is not None:
            img = self.transform(img) #是否进行transform
        label = torch.FloatTensor(label)
        return img,label  #return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
 
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
 
#根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
train_data_path = 'data/training_data/cropped/train.txt'
test_data_path  = 'data/training_data/cropped/test.txt'
train_data=MyDataset(datatxt=train_data_path, transform=transforms.ToTensor())
test_data =MyDataset(datatxt=test_data_path,  transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = Data.DataLoader(dataset=test_data,  batch_size=50)


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
