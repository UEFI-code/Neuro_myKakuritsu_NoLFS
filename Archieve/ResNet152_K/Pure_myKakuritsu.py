import torch
import torch.nn as nn
from torch.nn import functional as F
from myKakuritsu_Drv import myKakuritsu_Linear_Obj as myKakuritsu_Linear_Obj

class PureKakuritsu(nn.Module):
    def __init__(self, inshape = 2048, outshape = 1000):
        super(PureKakuritsu, self).__init__()
        self.relu = nn.ReLU(True)
        self.li1 = myKakuritsu_Linear_Obj(inshape, 1000, 0.5)
        self.li2 = myKakuritsu_Linear_Obj(1000, outshape, 0.5)

    def forward(self, x):
        x = self.li1(x)
        x = self.relu(x)
        x = self.li2(x)
        return x
