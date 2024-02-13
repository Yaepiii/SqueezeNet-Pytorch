import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms  # 将数据集转化为张量
from torchvision import datasets    # 下载数据集用
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
from SqueezeNet.SqueezeNet import SqueezeNet

class Trainer:
    def __init__(self, opt, model, train_loader):
        self.opt = opt
        self.model = model
        self.train_loader = train_loader
        self.losses = []

    def train(self):
        loss_fn = nn.CrossEntropyLoss()
        learning_rate = opt.lr
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=opt.m,
            weight_decay=opt.wd
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        for epoch in tqdm(range(opt.epoch)):
            scheduler.step()
            for i, (X, Y) in enumerate(self.train_loader):
                X, Y = X.to('cuda:0'), Y.to('cuda:0')
                Pred = self.model(X)
                loss = loss_fn(Pred, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % opt.epoch == 0:
                    self.losses.append(loss.item())
                    print('训练误差为: {:.4f}'.format(loss.item()))
    def printParamters(self):
        # 打印模型的 state_dict
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
    def saveParameters(self, root='./'):
        try:
            os.makedirs(root)
        except OSError:
            pass
        # state_dict()表示只保存训练好的权重
        torch.save(self.model.state_dict(), root + 'squeeze_model_' + 'version' + str(opt.version) + '_epoch' + str(opt.epoch) + '.pt')

    def plotLoss(self):
        Fig = plt.figure()
        plt.plot(range(len(self.losses)), self.losses)
        plt.show()

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--m', type=float, default=0.09, help='momentum')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--epoch', type=int, default=10, help='train epoch')
parser.add_argument('--version', type=int, default=1, help='squeezenet version(1/2)')

opt = parser.parse_args()

# 控制终端字符颜色
# \033[显示方式;前景色;背景色m + 内容 + 结尾部分：\033[0m
blue = lambda x: '\033[94m' + x + '\033[0m'

# 制作数据集

# 数据集转换参数
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomCrop(224),
    transforms.Resize(224),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载训练集与测试集
train_data = datasets.CIFAR10(
    root='./CIFAR10/',
    train=True,         # 是 train 集
    download=True,      # 如果该路径没有该数据集，就下载
    transform=transform # 数据集转换参数
)

test_data = datasets.CIFAR10(
    root='./CIFAR10/',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=opt.batch_size)

# 创建模型实例
model = SqueezeNet(version=opt.version).to('cuda:0')
model = model.train()

# 创建训练器并训练
trainer = Trainer(opt, model, train_loader)
trainer.train()

# 保存模型参数
trainer.saveParameters(root='parameter/')

# 可视化训练损失
trainer.plotLoss()

# state_dict就是一个简单的Python dictionary，其功能是将每层与层的参数张量之间一一映射

# 加载参数到GPU
state_dict = torch.load('parameter/' + 'squeeze_model_' + 'version' + str(opt.version) + '_epoch' + str(opt.epoch) + '.pt', map_location=torch.device('cuda:0'))
model.load_state_dict(state_dict)

# 打印模型参数
trainer.printParamters()

correct = 0
total = 0

model = model.eval()
with torch.no_grad():
    for i, (X, Y) in enumerate(test_loader):
        X, Y = X.to('cuda:0'), Y.to('cuda:0')
        Pred = model(X)
        _, predicted = torch.max(Pred.data, dim=1)
        correct += torch.sum((predicted==Y))
        total += Y.size(0)
        print(f'第 {blue(str(i))} 批测试精准度: {100*correct/total} %')


