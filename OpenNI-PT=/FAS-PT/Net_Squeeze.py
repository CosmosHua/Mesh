# coding:utf-8
# !/usr/bin/python3

import torch
import torch.nn as nn


# Ref: https://github.com/moskomule/senet.pytorch
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


LK = 5 # (negative_slope, inplace)
ReLU = nn.LeakyReLU(1/LK, True) if LK else nn.ReLU6(True)
################################################################################
class Fire(nn.Module): # inc->mid->(expand1+expand3)
    def __init__(self, inc, mid, expand1, expand3):
        '''
        self.squeeze = nn.Sequential(nn.Conv2d(inc, oup, 1), ReLU)
        self.expand1 = nn.Sequential(nn.Conv2d(oup, expand1, 1), ReLU)
        self.expand3 = nn.Sequential(nn.Conv2d(oup, expand3, 3, padding=1), ReLU)
        '''
        super(Fire, self).__init__() # BatchNorm accelerate convergence
        self.squeeze = nn.Sequential(nn.Conv2d(inc, mid, kernel_size=1),
                                     nn.BatchNorm2d(mid), ReLU)
        self.branch1 = nn.Sequential(nn.Conv2d(mid, expand1, kernel_size=1),
                                     nn.BatchNorm2d(expand1), ReLU)
        self.branch3 = nn.Sequential(nn.Conv2d(mid, expand3, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(expand3), ReLU)


    def forward(self, x): # spatial dimension unchanged
        '''
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], 1)
        '''
        x = self.squeeze(x) # x=(B,C,H,W), squeeze: C->mids
        return torch.cat([self.branch1(x), self.branch3(x)], dim=1)


# HW = [1+(HW+2*pad-dilation*(kernel-1)-1)//stride]
################################################################################
class SqueezeNet(nn.Module): # ver=1.1, hw=224
    def __init__(self, inc=3, cls=1000):
        super(SqueezeNet, self).__init__()
        cls = cls if type(cls)==int else len(cls)
        
        conv_init = nn.Conv2d(inc, 64, kernel_size=3, stride=2)
        self.features = nn.Sequential( # 224*224*inc->13*13*512
            conv_init, nn.BatchNorm2d(64), ReLU,                   # ->111*111*64
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->55*55*64
            Fire(64, 16, 64, 64), Fire(64*2, 16, 64, 64),          # ->55*55*(64+64)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->27*27*128
            Fire(64*2, 32, 128, 128), Fire(128*2, 32, 128, 128),   # ->27*27*(128+128)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->13*13*256
            Fire(128*2, 48, 192, 192), Fire(192*2, 48, 192, 192),  # ->13*13*(192+192)
            Fire(192*2, 64, 256, 256), Fire(256*2, 64, 256, 256) ) # ->13*13*(256+256)
        
        conv_last = nn.Conv2d(256*2, cls, kernel_size=1) # ->13*13*cls
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), conv_last,
            nn.BatchNorm2d(cls), ReLU, nn.AdaptiveAvgPool2d(1) )
        
        for m in self.modules(): # initialization
            if isinstance(m, nn.Conv2d): # initialize convs
                if m is not conv_last: nn.init.kaiming_uniform_(m.weight)
                else: nn.init.xavier_uniform_(m.weight) # xavier_normal_
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): # initialize linear
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d): # initialize norm
                m.weight.data.fill_(1); m.bias.data.zero_()


    def forward(self, x): # x=(B,C,H->1,W->1)
        x = self.classifier(self.features(x))
        return x.view(x.size(0), -1)


# HW = [1+(HW+2*pad-dilation*(kernel-1)-1)//stride]
PWConv = lambda i,o,b=0: nn.Conv2d(i, o, 1, 1, 0, bias=b)
DWConv = lambda c,k=3,s=1,p=0,b=0: nn.Conv2d(c, c, k, s, p, groups=c, bias=b)
################################################################################
class SqueezeNet2(nn.Module): # ver=1.2, hw=224
    def __init__(self, inc=3, cls=1000):
        super(SqueezeNet2, self).__init__()
        cls = cls if type(cls)==int else len(cls)
        
        conv_init = nn.Conv2d(inc, 64, kernel_size=3, stride=2)
        self.features = nn.Sequential( # 224*224*inc->13*13*512
            conv_init, nn.BatchNorm2d(64), ReLU,                   # ->111*111*64
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->55*55*64
            Fire(64, 16, 64, 64), Fire(64*2, 16, 64, 64),          # ->55*55*(64+64)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->27*27*128
            Fire(64*2, 32, 128, 128), Fire(128*2, 32, 128, 128),   # ->27*27*(128+128)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->13*13*256
            Fire(128*2, 48, 192, 192), Fire(192*2, 48, 192, 192),  # ->13*13*(192+192)
            Fire(192*2, 64, 256, 256), Fire(256*2, 64, 256, 256) ) # ->13*13*(256+256)
        
        # 13*13*512->13*13*cls->1*1*cls
        #self.classifier = nn.Sequential(nn.Dropout(p=0.5),
        #    PWConv(512,cls), nn.BatchNorm2d(cls), ReLU, DWConv(cls,13) )
        
        # 13*13*512->5*5*512->5*5*cls->1*1*cls
        #self.classifier = nn.Sequential(nn.Dropout(p=0.5), DWConv(512,5,2),
        #    PWConv(512,cls), nn.BatchNorm2d(cls), ReLU, DWConv(cls,5) )
        
        # 13*13*512->6*6*512->6*6*cls->1*1*cls
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), DWConv(512,3,2),
            PWConv(512,cls), nn.BatchNorm2d(cls), ReLU, DWConv(cls,6))
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
