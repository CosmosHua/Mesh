# coding:utf-8
# !/usr/bin/python3
# https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py

import torch
import torch.nn as nn


ReLU = nn.ReLU(inplace=True)
PWConv = lambda i,o,b=0: nn.Conv2d(i, o, 1, 1, 0, bias=b)
DWConv = lambda c,k=3,s=1,p=0,b=0: nn.Conv2d(c, c, k, s, p, groups=c, bias=b)
# nn.Conv2d(inp, oup, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True)
################################################################################
def conv1_bn(inp, oup): return [PWConv(inp, oup), nn.BatchNorm2d(oup), ReLU]


def conv3_bn(inp, oup, stride):
    return [nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), ReLU]


def dwconv3_bn(inp, stride):
    return [nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp)]


def channel_shuffle(x, group=2):
    B, C, H, W = x.size(); assert C%group==0
    x = x.view(B, group, C//group, H, W) # reshape->group
    x = torch.transpose(x, 1, 2).contiguous() # transpose
    return x.view(B, -1, H, W) # flatten


# HW = [1+(HW+2*pad-dilation*(kernel-1)-1)//stride]
################################################################################
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        assert 1<=stride<=3
        self.stride = stride; branch = oup//2
        assert (stride!=1) or (inp==branch<<1)
        
        if stride>1:
            self.branch1 = nn.Sequential(*dwconv3_bn(inp, stride), *conv1_bn(inp, branch) )
        self.branch2 = nn.Sequential(*conv1_bn(inp if stride>1 else branch, branch),
                                    *dwconv3_bn(branch, stride), *conv1_bn(branch, branch) )


    def forward(self, x):
        if self.stride==1:
            x1, x2 = x.chunk(2, dim=1) # x.split(2,dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else: # self.stride>1
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


S2I = lambda x: [int(i) for i in x.strip("()[]").split(",") if i.strip().isdigit()]
################################################################################
class ShuffleNet(nn.Module):
    def __init__(self, inp=3, cls=1000, R=1.0):
        super(ShuffleNet, self).__init__()
        if R==0.5:   stage, oup = (4,8,4), (24, 48, 96, 192, 1024)
        elif R==1.0: stage, oup = (4,8,4), (24, 116, 232, 464, 1024)
        elif R==1.5: stage, oup = (4,8,4), (24, 176, 352, 704, 1024)
        elif R==2.0: stage, oup = (4,8,4), (24, 244, 488, 976, 2048)
        else: stage = S2I(input("stage:\t")); oup = S2I(input("oup:\t"))
        assert len(stage)==3 and len(oup)==5
        
        cls = cls if type(cls)==int else len(cls); mid = oup[0]
        self.conv1 = nn.Sequential(nn.Conv2d(inp, mid, 3, 2, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True) )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        inp = mid
        name = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, num, mid in zip(name, stage, oup[1:]):
            seq = [InvertedResidual(inp, mid, 2)]
            for i in range(num-1): seq += [InvertedResidual(mid, mid, 1)]
            inp = mid; setattr(self, name, nn.Sequential(*seq))

        mid = oup[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True) )
        self.fc = nn.Linear(mid, cls)


    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.stage3(self.stage2(x))
        x = self.conv5(self.stage4(x))
        x = x.mean([2, 3]) # globalpool
        return self.fc(x)


################################################################################
