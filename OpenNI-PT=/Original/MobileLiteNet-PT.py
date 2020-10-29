import math
import torch
import torch.nn as nn
import numpy as np


# Ref: https://github.com/moskomule/senet.pytorch
################################################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction), nn.ReLU(inplace=True),
                nn.Linear(channel//reduction, channel), nn.Sigmoid() )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Ref: https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
# https://github.com/CosmosHua/FeatherNets/blob/regression/models/MobileLiteNet.py
################################################################################
class InvertedResidual(nn.Module):
    expand = 6 # attribute of class(not instance)
    def __init__(self, inp, oup, stride=1, down=None):
        super(InvertedResidual, self).__init__()
        self.down = down; mid = inp*6
        ReLU = nn.ReLU6(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid), ReLU, # conv1
            nn.Conv2d(mid, mid, 3, stride, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(mid), ReLU, # conv2
            nn.Conv2d(mid, oup , kernel_size=1, bias=False), nn.BatchNorm2d(oup) )


    def forward(self, x):
        dx = self.down(x) if self.down else x
        return self.conv(x)+dx


DWConv = lambda c,k,s=1,p=0,b=0: nn.Conv2d(c, c, k, s, p, groups=c, bias=b)
################################################################################
class MobileLiteNet(nn.Module):
    def __init__(self, block, layers, cls=2, use_se=False):
        super(MobileLiteNet, self).__init__()
        ReLU = nn.ReLU6(inplace=True); net = []
        ch = [32, 16, 32, 48, 64]; self.mid = ch[1]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]), ReLU,
            nn.Conv2d(ch[0], ch[1], 1, bias=False), nn.BatchNorm2d(ch[1]), ReLU)
        
        for i in range(4):
            net += self._make_layer(block, ch[i+1], layers[i], stride=2)
            if use_se: net += [SELayer(ch[i+1])]
        self.conv2 = nn.Sequential(*net)
        
        self.final_DW = DWConv(ch[4], 4)
        self.linear = nn.Sequential(nn.Dropout(0.2), nn.Linear(ch[4]*16, cls))
        self._initialize_weights()


    def _make_layer(self, block, oup, num, stride=1):
        down = nn.Sequential(DWConv(self.mid, 3, stride, 1),
            nn.BatchNorm2d(self.mid), nn.Conv2d(self.mid, oup, 1, bias=False) )
        down = down if stride!=1 else None
        layers = [block(self.mid, oup, stride, down)]
        for i in range(1,num): layers += [block(oup, oup)]
        return layers # type=list


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear): # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01); m.bias.data.zero_()


    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.final_DW(x)
        return self.linear(x.view(x.size(0),-1))


################################################################################
def MobileLiteNets(cls=2, layers=54, use_se=True):
    if layers==54: layers = [4,4,6,3]
    elif layers==102: layers = [3,4,23,3]; use_se = False
    elif layers==105: layers = [4,4,23,3]; use_se = True
    elif layers==153: layers = [3,8,36,3]; use_se = False
    elif layers==156: layers = [4,8,36,3]; use_se = True
    else: assert len(layers)>3
    return MobileLiteNet(InvertedResidual, layers, cls, use_se)


################################################################################
