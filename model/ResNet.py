import torch
import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels !=  out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.shortcut(residual)
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, n, num_classes=10):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, 16, n)
        self.layer2 = self._make_layer(block, 16, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 32, 64, n, stride=2)

        self.fc = nn.Linear(4096, num_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, in_channels, out_channels, n, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride=stride))

        for _ in range(1, n):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet20(num_classes=10):
    return ResNet(BasicBlock, 3, num_classes)

def resnet32(num_classes=10):
    return ResNet(BasicBlock, 5, num_classes)

def resnet44(num_classes=10):
    return ResNet(BasicBlock, 7, num_classes)

def resnet56(num_classes=10):
    return ResNet(BasicBlock, 9, num_classes)

