import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=2, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Block(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = Block(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = Block(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = Block(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class DeformationClassifier(nn.Module):
    def __init__(self, n_channels, n_classes, num_pixels):
        super(DeformationClassifier, self).__init__()
        self.CNN = nn.Sequential(
                        nn.Conv2d(in_channels=n_channels, out_channels=12, kernel_size=5, padding=1, stride=1, bias=True),
                        nn.BatchNorm2d(12),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(in_channels=12, out_channels=48, kernel_size=5, padding=1, stride=1, bias=True),
                        nn.BatchNorm2d(48),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5, padding=1, stride=2, bias=True),
                        nn.BatchNorm2d(192),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(in_channels=192, out_channels=384, kernel_size=5, padding=1, stride=2, bias=True),
                        nn.BatchNorm2d(384),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2)
                        )
        self.flatten = nn.Flatten()
        self.MLP = nn.Sequential(
                        nn.Dropout(p=0.20),
                        nn.Linear(3840, 384),
                        nn.ReLU(),
                        nn.Dropout(p=0.20),
                        nn.Linear(384, 96),
                        nn.ReLU(),
                        nn.Dropout(p=0.20),
                        nn.Linear(96, n_classes)
                        )

    def forward(self, x):
        x = self.CNN(x)
        x = self.flatten(x)
        x = self.MLP(x)
        return x