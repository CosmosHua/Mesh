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


PWConv = lambda i,o: nn.Conv2d(i, o, 1, 1, 0, bias=False)
DWConv3 = lambda c,s: nn.Conv2d(c, c, 3, s, 1, groups=c, bias=False)
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


    def forward(self, x):
        if self.use_res: return self.conv(x)+x
        else:
            if not self.down: return self.conv(x)
            else: return self.conv(x)+self.down(x)


################################################################################
class FeatherNet(nn.Module):
    def __init__(self, inc=3, cls=2, hw=224, use_se=True, avgdown=False, widen=1):
        super(FeatherNet, self).__init__()
        assert hw%32==0; Block = InvertedResidual
        cls = cls if type(cls)==int else len(cls)
        self.use_se = use_se; self.avgdown = avgdown
        
        inverted_residual_setting = [    # t, c,  n, s
            [1, 16, 1, 2], [6, 32, 2, 2], [6, 48, 6, 2], [6, 64, 3, 2] ]
            # 112x112  ->   # 56x56    ->   # 14x14   ->   # 7x7
        
        inp = int(32*widen); last = int(1024*widen)
        features = [conv_bn(inc, inp, 2)] # first layer
        # build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            mid = int(c*widen)
            for i in range(n):
                down = None
                if i == 0:
                    if avgdown: down = down_bn_conv(inp, mid)
                    features.append(Block(inp, mid, s, expand=t, down=down))
                else:
                    features.append(Block(inp, mid, 1, expand=t, down=down))
                inp = mid
            if use_se: features.append(SELayer(inp)) # SELayer
        self.features = nn.Sequential(*features) # make nn.Sequential
        self.final_DW = nn.Sequential(DWConv3(inp,s=2)) # ->4x4*64
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


    def forward(self, x):
        x = self.features(x); x = self.final_DW(x)
        return x.view(x.size(0),-1)


################################################################################
FeatherNetA = lambda inc,cls: FeatherNet(inc, cls)
FeatherNetB = lambda inc,cls: FeatherNet(inc, cls, avgdown=True)
