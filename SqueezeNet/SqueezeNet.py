import torch
import torch.nn as nn
from torchinfo import summary

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, version=1):
        super(Fire, self).__init__()
        self.in_channels = in_channels
        self.version = version
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.bn_squeeze = nn.BatchNorm2d(squeeze_channels)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(expand1x1_channels + expand3x3_channels)

    def forward(self, x):
        # x1 = self.squeeze_activation(self.squeeze(x))
        # e1 = self.expand1x1(x1)
        # e2 = self.expand3x3(x1)
        x1 = self.activation(self.bn_squeeze(self.squeeze(x)))
        e1 = self.expand1x1(x1)
        e2 = self.expand3x3(x1)
        y = torch.cat([e1, e2], 1)
        if self.version == 1:
            y = self.activation(self.bn_out(y))
        elif self.version == 2:
            # y = self.expand_activation(y + x)
            y = self.activation(self.bn_out(y + x))
        return y

class SqueezeNet(nn.Module):
    def __init__(self, version=1, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.version = version
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire5 = Fire(256, 32, 128, 128)
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)
        self.conv10 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        '''
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),    # 自适应平均池化，指定输出（H，W）
        )

    def forward(self, x):
        '''
        x = self.conv1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.mp(self.fire4(x))
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.mp(self.fire8(x))
        x = self.fire9(x)
        x = self.fire3(x) + x
        x = self.mp(self.fire4(x))
        x = self.conv10(x)
        '''
        x = self.net(x)
        x = self.classifier(x)
        y = torch.flatten(x, 1)
        return y


def test():
    test_model = SqueezeNet(version=2, num_classes=10)
    y = test_model(torch.randn(1, 3, 224, 224))
    print(y.size())
    summary(test_model, (1, 3, 224, 224))

if __name__ == '__main__':
    test()


