# coding:utf-8
# !/usr/bin/python3

import torch
import torch.nn as nn
from DataAug import inc


# HW = [1+(HW+2*pad-dilation*(kernel-1)-1)//stride]
ReLU = nn.LeakyReLU(0.2, True); BN = lambda c: nn.BatchNorm2d(c)
PWConv = lambda i,o,g=1,b=0: nn.Conv2d(i, o, 1, 1, 0, groups=g, bias=b)
DWConv = lambda c,k=3,s=1,p=0,b=0: nn.Conv2d(c, c, k, s, p, groups=c, bias=b)
GPConv = lambda i,o,k=3,s=1,p=0,g=1,b=0: nn.Conv2d(i, o, k, s, p, groups=g, bias=b)
# nn.Conv2d(inp, oup, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True)
################################################################################
def Shuffle(x, group): # regroup
    B,C,H,W = x.size(); assert C%group==0
    x = x.view(B, group, C//group, H, W) # reshape
    x = torch.transpose(x, 1, 2).contiguous() # transpose
    return x.view(B, -1, H, W) # reshape


def CShuffle(x): # concatenate->regroup
    B,C,H,W = x[0].size(); group = len(x)
    x = torch.cat(x, dim=1).view(B, group, C, H, W) # reshape
    x = torch.transpose(x, 1, 2).contiguous() # transpose
    return x.view(B, -1, H, W) # reshape


################################################################################
class SELayer(nn.Module):
    def __init__(self, C, down=8):
        super(SELayer, self).__init__()
        mid = max(1, C//down)
        self.se = nn.Sequential(nn.Linear(C,mid),
            ReLU, nn.Linear(mid,C), nn.Sigmoid() )
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.gap(x).view(B, C)
        y = self.se(y).view(B, C, 1, 1)
        return x * y


################################################################################
class DWS(nn.Module): # DWConv->Shuffle
    def __init__(self, inp, oup, s=2, K=(3,5), N=2):
        super(DWS, self).__init__() # hw=(hw+2p-k)//s+1=(hw-1)//s+1
        dwbn = []; assert inp%inc==0 and oup%inc==0; C = N*len(K)*inp
        for k in K*N: dwbn += [nn.Sequential(DWConv(inp,k,s,k//2), BN(inp))]
        self.dwbn = nn.ModuleList(dwbn); self.SE = SELayer(oup, inc)
        self.pwbn = nn.Sequential(PWConv(C,oup,g=inc), BN(oup))


    def forward(self, x): # x=(B,C,H,W)->C=oup
        x = CShuffle([op(x) for op in self.dwbn])
        return self.SE(self.pwbn(ReLU(x)))


################################################################################
class DSGS(nn.Module): # Depthwise Separable+Group Conv->Shuffle
    def __init__(self, inp, oup, s=2, K=(3,5), N=1):
        super(DSGS, self).__init__()
        dgbn = []; G = len(K)+(s>1); assert oup%(G*inc)==0; C = oup//G
        for i in range(N): # hw=(hw+2p-k)//s+1=(hw-1)//s+1
            if s>1: dgbn += [nn.Sequential(GPConv(inp,C,k=3,s=s,p=1,g=inc), BN(C))]
            for k in K: dgbn += [nn.Sequential(
                DWConv(inp,k,s,k//2), BN(inp), ReLU, PWConv(inp,C,g=inc), BN(C) )]
        self.dgbn = nn.ModuleList(dgbn); self.SE = SELayer(oup); self.G = G
        self.dsbn = nn.Sequential(
            DWConv(inp,k=3,s=s,p=1), BN(inp), ReLU, PWConv(inp,oup,g=1), BN(oup) )


    def forward(self, x): # x=(B,C,H,W)->C=oup
        P = [op(x) for op in self.dgbn]; y = self.dsbn(x); G = self.G
        for i in range(0,len(P),G): y += CShuffle(P[i:i+G])
        return self.SE(y) # Residual


################################################################################
class DSGS2(nn.Module): # Depthwise Separable+Group Conv->Shuffle
    def __init__(self, inp, oup, s=2, K=(3,5), N=1):
        super(DSGS2, self).__init__()
        dgbn = []; G = N*(len(K)+(s>1)); assert oup%(G*inc)==0; C = oup//G
        for i in range(N): # hw=(hw+2p-k)//s+1=(hw-1)//s+1
            if s>1: dgbn += [nn.Sequential(GPConv(inp,C,k=3,s=s,p=1,g=inc), BN(C))]
            for k in K: dgbn += [nn.Sequential(
                DWConv(inp,k,s,k//2), BN(inp), ReLU, PWConv(inp,C,g=inc), BN(C) )]
        self.dgbn = nn.ModuleList(dgbn); self.SE = SELayer(oup); self.G = G
        self.dsbn = nn.Sequential(
            DWConv(inp,k=3,s=s,p=1), BN(inp), ReLU, PWConv(inp,oup,g=1), BN(oup) )


    def forward(self, x): # x=(B,C,H,W)->C=oup
        y = CShuffle([op(x) for op in self.dgbn])
        return self.SE(self.dsbn(x) + y) # Residual


################################################################################
class GroupNet(nn.Module): # hw=112
    def __init__(self, inp=inc, cls=100, R=1.0):
        super(GroupNet, self).__init__(); Block = DSGS
        cls = cls if type(cls)==int else len(cls); K = (3,5,7)
        self.features = nn.Sequential(
            DWS(inp, 72, s=2, K=K, N=2),        # hw=112->56, oup%inc
            Block( 72,  72, s=1, K=K, N=1),     # hw=56->56, oup%(1*3*inp)
            Block( 72, 144, s=2, K=K, N=1),     # hw=56->28, oup%(1*4*inp)
            Block(144, 144, s=1, K=K, N=1),     # hw=28->28, oup%(1*3*inp)
            Block(144, 288, s=2, K=K, N=1),     # hw=28->14, oup%(1*4*inp)
            Block(288, 384, s=2, K=K, N=1))     # hw=14-> 7, oup%(1*4*inp)
        self.classifier = nn.Sequential(nn.Dropout(p=0.8),
            PWConv(384,cls), BN(cls), ReLU, DWConv(cls,7) )
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules(): # initialization
            if isinstance(m, nn.Conv2d): # initialize all convs
                if m.groups<2: nn.init.kaiming_uniform_(m.weight)
                else: nn.init.xavier_uniform_(m.weight) # xavier_normal_
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); m.bias.data.zero_()


    def forward(self, x): # x=(B,C,H->1,W->1)
        x = self.classifier(self.features(x))
        return x.view(x.size(0), -1)


################################################################################
