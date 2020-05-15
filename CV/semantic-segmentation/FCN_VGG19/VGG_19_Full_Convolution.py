import math
import torch
import torch.nn as nn
import numpy as np

# 定义 Block 组件，该组件是一层卷积 + BN + ReLU
class Block(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


# make_layers 给定输入图像 channel 数目，和要经过每层 layer 的通道数据
# 返回多个 layer， layer_list[64, 64]
def make_layers(in_channels, layer_list):
    layers = []
    for v in layer_list:
        layers += [Block(in_channels, v)]
        in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list)

    def forward(self, x):
        out = self.layer(x)
        return out


class VGG_19_Full_Convolution(nn.Module):
    def __init__(self, n_class):
        super(VGG_19_Full_Convolution, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = nn.Conv2d(512, 4096, 7)  # padding=0
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score = nn.Conv2d(4096, n_class, 1)

    def forward(self, x):

        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))

        f6 = self.drop6(self.relu6(self.fc6(f5)))

        result = self.score(self.drop7(self.relu7(self.fc7(f6))))

        return result

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    model = VGG_19_Full_Convolution(21)
    model.eval()
    y = model(x)
    print(y.shape)