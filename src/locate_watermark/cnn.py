import torch
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
import torch.nn as nn
#设置随机种子以便复现
torch.manual_seed(1)
EPOCH=1
BATCH_SIZE=64
LR=0.001
#如果已经下载好改为False
DOWNLOAD_MINIST= False
#获取手写数字训练集
train_data = torchvision.datasets.MNIST(
    # 保存地点
    root="./minist/",
    # 是否是训练集
    train=True,
    # 转换 PIL.Image or numpy.ndarray 成torch.FloatTensor (C,H,W), 训练的时候 normalize 成 [0.0, 1.0]区间
    transform=torchvision.transforms.ToTensor(),
    # 是否下载
    download=DOWNLOAD_MINIST)
# 可视化查看一下数据图片

train,label = train_data[0]
plt.imshow(train.squeeze(), cmap='gray')
plt.savefig("test.jpg")
test_data = torchvision.datasets.MNIST(
    # 保存地点
    root="./minist/",
    # 是否是训练集
    train=False
)
# 批训练(32,1,28,28)
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
# 由原来的(60000,28,28)变为(60000,1,28,28)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]
print(test_x.size())
print(test_y.size())
# 若存在cuda环境，即可使用注释代码
test_x = test_x.cuda()
test_y = test_y.cuda()



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #输入形状为(1,128,128)，输出形状为(16,64,64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,   # 输入通道
                      out_channels=16, # 输出通道
                      kernel_size=5,   # 卷积核大小
                      stride=1,        # 步幅
                      padding=2),      # 填充
            # 此时输出形状为(16,128,128)
            nn.ReLU(),                 # 激活函数
            # 最大池化，核大小为2，此时输出形状(16,64,64)
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2), # 输出形状(32,64,64)
            nn.ReLU(),
            nn.MaxPool2d(2)         #输出形状(32,32,32)
        )
        #全连接网络，输出10个类别
        self.out = nn.Linear(32*32*32,4)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将卷积层展平，才能输入全连接网络
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output


cnn = CNN()
# 若存在cuda环境，即可使用注释代码
cnn = cnn.cuda()
# 优化器
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
# 损失函数，分类问题
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    # 迭代运行
    for step, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        output = cnn(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(step % 50 == 0):
            # 可以单独进行模型的测试
            test_output = cnn(test_x)
            # 1代表索引列，因为刚好匹配到0-9，获取概率高的
            # pre_y = torch.max(test_output, 1)[1].data.squeeze()
            
            pre_y = torch.max(test_output, 1)[1]
            # 获取准确率
            accurary = float((pre_y == test_y).sum()) / float(test_y.size(0))
            print('Epoch: ',epoch, '| train loss: %.4f' % loss.data, '| test accurary: %.2f' % accurary)
# 最后可以模型保存
test_output = cnn(test_x[:10])
print(test_output)
print(torch.max(test_output, 1))
print(torch.max(test_output, 1)[1])
pred_y = torch.max(test_output, 1)[1]
print(pred_y, 'prediction number')
print(test_y[:10].squeeze(), 'real number')