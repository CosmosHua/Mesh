# coding:utf-8
# !/usr/bin/python3
# https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.model_zoo import load_url


ReLU = nn.ReLU(inplace=True)
################################################################################
class Fire(nn.Module):
    def __init__(self, inc, oup, expand1, expand3):
        super(Fire, self).__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(inc, oup, 1), ReLU)
        self.expand1 = nn.Sequential(nn.Conv2d(oup, expand1, 1), ReLU)
        self.expand3 = nn.Sequential(nn.Conv2d(oup, expand3, 3, padding=1), ReLU)


    def forward(self, x):
        x = self.squeeze(x) # x=(B,C,H,W)
        return torch.cat([self.expand1(x), self.expand3(x)], 1)


################################################################################
class SqueezeNet(nn.Module):
    def __init__(self, version=1.0, cls=1000):
        super(SqueezeNet, self).__init__()
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2), ReLU,
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64), Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128), Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192), Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256) )
        elif version == 1.1:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2), ReLU,
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64), Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128), Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256), Fire(512, 64, 256, 256) )
        else: raise ValueError("Unsupported SqueezeNet version %.1f" % version)
        
        self.cls = cls # different initialization for final_conv
        final_conv = nn.Conv2d(512, cls, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
            final_conv, ReLU, nn.AdaptiveAvgPool2d((1,1)) )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is not final_conv: init.kaiming_uniform_(m.weight)
                else: init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.classifier(self.features(x))
        return x.view(x.size(0), self.cls)


model_url = "https://download.pytorch.org/models/"
pre_train = {'squeezenet1.0': model_url+'squeezenet1_0-a815701f.pth',
             'squeezenet1.1': model_url+'squeezenet1_1-f364aa15.pth'}
################################################################################
def squeezenet1_0(pretrained=False, **kwargs):
    """SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained: model.load_state_dict(load_url(pre_train['squeezenet1.0']))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    """SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained: model.load_state_dict(load_url(pre_train['squeezenet1.1']))
    return model


################################################################################
