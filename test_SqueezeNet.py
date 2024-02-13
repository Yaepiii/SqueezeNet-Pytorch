import torch
import torch.nn as nn
from torchvision import transforms  # 将数据集转化为张量
from torchvision import datasets    # 下载数据集用
from torch.utils.data import DataLoader
import argparse
from SqueezeNet.SqueezeNet import SqueezeNet

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=10, help='train epoch')
parser.add_argument('--version', type=int, default=1, help='squeezenet version(1/2)')

opt = parser.parse_args()

# 控制终端字符颜色
# \033[显示方式;前景色;背景色m + 内容 + 结尾部分：\033[0m
blue = lambda x: '\033[94m' + x + '\033[0m'

# 数据集转换参数
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomCrop(224),
    transforms.Resize(224),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


test_data = datasets.CIFAR10(
    root='./CIFAR10/',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_data, shuffle=True, batch_size=opt.batch_size)

model = SqueezeNet(version=opt.version).to('cuda:0')

# 加载参数到GPU
state_dict = torch.load('parameter/' + 'squeeze_model_' + 'version' + str(opt.version) + '_epoch' + str(opt.epoch) + '.pt', map_location=torch.device('cuda:0'))
model.load_state_dict(state_dict)

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