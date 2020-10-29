import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


################################################################################
# nn.Conv2d: (in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, padding=1, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU(inplace=True) )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU(inplace=True) )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width) # reshape
    x = torch.transpose(x, 1, 2).contiguous() # transpose
    x = x.view(batchsize, -1, height, width) # flatten
    return x


################################################################################
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        if self.benchmodel:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True) )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True) )
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True) )


    def forward(self, x):
        if self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = torch.cat((x1, self.banch2(x2)), 1)
        else: # concatenate along channel axis
            out = torch.cat((self.banch1(x), self.banch2(x)), 1)
        return channel_shuffle(out, 2)


################################################################################
class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(ShuffleNetV2, self).__init__()
        assert input_size % 32 == 0
        
        if width_mult == 0.25:   self.out_channels = [24, 24,  48,  96,  512]
        elif width_mult == 0.33: self.out_channels = [24, 32,  64, 128,  512]
        elif width_mult == 0.5: self.out_channels = [24,  48,  96, 192, 1024]
        elif width_mult == 1.0: self.out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5: self.out_channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0: self.out_channels = [24, 224, 488, 976, 2048]
        else: raise ValueError("Unsupported for 1x1 Grouped Convolutions!\n")

        # building first layer
        input_channel = self.out_channels[0]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        self.stage_repeats = [4, 8, 4]
        # building inverted residual blocks
        for stage, repeat in enumerate(self.stage_repeats)
            output_channel = self.out_channels[stage+1]
            for i in range(repeat): # inp, oup, stride, benchmodel
                self.features.append(InvertedResidual(input_channel, output_channel, 2, i>0))
                input_channel = output_channel
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))
    
        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.out_channels[-1], n_class))


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.out_channels[-1])
        x = self.classifier(x)
        return x


################################################################################
def ShuffleNet2(width_mult=1.0):
    return ShuffleNetV2(width_mult=width_mult)


################################################################################
if __name__ == "__main__":
    """Testing"""
    model = ShuffleNetV2()
    print(model)
