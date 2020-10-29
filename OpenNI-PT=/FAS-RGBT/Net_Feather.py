# coding:utf-8
# !/usr/bin/python3

import torch, math
import torch.nn as nn


# Ref: https://github.com/moskomule/senet.pytorch
# Ref: https://github.com/tonylins/pytorch-mobilenet-v2
################################################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction), nn.ReLU(True),
            nn.Linear(channel//reduction, channel), nn.Sigmoid() )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# HW = [1+(HW+2*pad-dilation*(kernel-1)-1)//stride]
PWConv = lambda i,o,b=0: nn.Conv2d(i, o, 1, 1, 0, bias=b)
DWConv3 = lambda c,s,b=0: nn.Conv2d(c, c, 3, s, 1, groups=c, bias=b)
################################################################################
def conv1_bn(inp, oup):
    return nn.Sequential(PWConv(inp, oup), nn.BatchNorm2d(oup), nn.ReLU6(True))


def conv3_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU6(True) )


def down_bn_conv(inp, oup, stride=2):
    return nn.Sequential(nn.AvgPool2d(2, stride), nn.BatchNorm2d(inp), PWConv(inp, oup))


################################################################################
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand, down=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1,2]
        self.use_res = (stride==1 and inp==oup)
        mid = round(inp*expand)
        if expand==1:
            self.conv = nn.Sequential( # DWConv->PWConv_Linear
                DWConv3(mid, stride), nn.BatchNorm2d(mid), nn.ReLU6(True),
                PWConv(mid, oup), nn.BatchNorm2d(oup) )
        else:
            self.conv = nn.Sequential( # PWConv->DWConv->PWConv_Linear
                PWConv(inp, mid), nn.BatchNorm2d(mid), nn.ReLU6(True),
                DWConv3(mid, stride), nn.BatchNorm2d(mid), nn.ReLU6(True),
                PWConv(mid, oup), nn.BatchNorm2d(oup) )
        self.down = down_bn_conv(inp, oup, stride) if down else None


    def forward(self, x): # y = (B,C,H,W)
        if self.use_res: return self.conv(x)+x
        else:
            if not self.down: return self.conv(x)
            else: return self.conv(x)+self.down(x)


################################################################################
class FeatherNet(nn.Module): # hw=224
    def __init__(self, inc=3, cls=2, use_se=True, down=None, widen=1.0):
        super(FeatherNet, self).__init__()
        cls = cls if type(cls)==int else len(cls)
        self.inc = (inc,) if type(inc)==int else inc
        
        # (repeat,channel,stride,expand): 112*112->56*56->14*14->7*7
        Block_Setting = [(1,16,2,1), (2,32,2,6), (6,48,2,6), (3,64,2,6)]
        Block = InvertedResidual; mid = int(32*widen) # 224*224->112*112
        feature = [tuple(conv3_bn(i,mid,stride=2) for i in self.inc)]
        for N,C,S,T in Block_Setting:
            C = int(C*widen)
            for i in range(N):
                if i>0: feature += [Block(C, C, 1, expand=T, down=None)]
                else: feature += [Block(mid, C, S, expand=T, down=down)]
            if use_se: feature += [SELayer(C)] # append SELayer
            mid = C # finally C=int(64*widen), feature_map=7*7*C
        self.branch = [nn.Sequential(i,*feature[1:]) for i in feature[0]]
        
        C *= len(self.branch); oup = int(4*4*C)
        self.final_DW = nn.Sequential(DWConv3(C,s=2)) # branch concat
        self.linear = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(oup,cls))
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2/n))
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear): # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01); m.bias.data.zero_()


    def forward(self, x): # y = (B,C,H,W)
        x = torch.split(x, self.inc, dim=1) # channel
        x = [b.to(i.device)(i) for i,b in zip(x,self.branch)]
        x = self.final_DW(torch.cat(x, dim=1))
        return self.linear(x.view(x.size(0), -1))


################################################################################
FeatherNetA = lambda inc,cls: FeatherNet(inc, cls)
FeatherNetB = lambda inc,cls: FeatherNet(inc, cls, down=True)
