import torch
import torch.nn as nn
from torch.nn import functional as F
from myLinear_Drv import myLinear as myLinear

class PureDrop(nn.Module):
    def __init__(self, inshape = 2048, outshape = 1000):
        super(PureDrop, self).__init__()
        self.relu = nn.ReLU(True)
        self.drop = nn.Dropout(p = 0.5)
        self.li1 = myLinear(inshape, 1000)
        self.li2 = myLinear(1000, outshape)

    def forward(self, x):
        x = self.li1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.li2(x)
        x = self.drop(x)
        return x
