import torch
import torch.nn as nn
from torch.nn import functional as F
from myKakuritsu_Drv import myKakuritsu_Linear_Obj as myKakuritsu_Linear_Obj

class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.down(x)
        return out


class ResNet152(nn.Module):
    def __init__(self, classes_num = 1000):
        super(ResNet152, self).__init__()

        self.Kakuritsu1 = myKakuritsu_Linear_Obj(2048, 1000, 0.5)
        self.Kakuritsu2 = myKakuritsu_Linear_Obj(1000, classes_num, 0.5)
        
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1_shortcut = DownSample(64, 256, 1)
        self.layer2_shortcut = DownSample(256, 512, 2)
        self.layer3_shortcut = DownSample(512, 1024, 2)
        self.layer4_shortcut = DownSample(1024, 2048, 2)

        self.layer1_first = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer1_next = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256)
        )

        self.layer2_first = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512)
        )
        self.layer2_next = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512)
        )

        self.layer3_first = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.layer3_next = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024)
        )

        self.layer4_first = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048)
        )
        self.layer4_next = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
                self.Kakuritsu1,
                nn.ReLU(True),
                self.Kakuritsu2,
        )

    def forward(self, x):
        out = self.pre(x)

        layer1_identity = self.layer1_shortcut(out)
        out = self.layer1_first(out)
        out = F.relu(out + layer1_identity, inplace=True)

        for i in range(2):
            identity = out
            out = self.layer1_next(out)
            out = F.relu(out + identity, inplace=True)

        layer2_identity = self.layer2_shortcut(out)
        out = self.layer2_first(out)
        out = F.relu(out + layer2_identity, inplace=True)

        for i in range(7):
            identity = out
            out = self.layer2_next(out)
            out = F.relu(out + identity, inplace=True)

        layer3_identity = self.layer3_shortcut(out)
        out = self.layer3_first(out)
        out = F.relu(out + layer3_identity, inplace=True)

        for i in range(35):
            identity = out
            out = self.layer3_next(out)
            out = F.relu(out + identity, inplace=True)

        layer4_identity = self.layer4_shortcut(out)
        out = self.layer4_first(out)
        out = F.relu(out + layer4_identity, inplace=True)

        for i in range(2):
            identity = out
            out = self.layer4_next(out)
            out = F.relu(out + identity, inplace=True)

        out = self.avg_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out

