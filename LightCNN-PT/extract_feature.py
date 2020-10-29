'''
    implement the feature extractions for light CNN
    @author: Alfred Xiang Wu @date: 2017.07.04
    @modifier: CosmosHua @date: 2019/01/09
'''

# coding:utf-8
# !/usr/bin/python3

import numpy as np
import os, cv2, time
import argparse, shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2


################################################################################
parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint, default: none')
parser.add_argument('--model', default='', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29, LightCNN-29v2')
parser.add_argument('--img_dir', default='', type=str, metavar='PATH',
                    help='root path of face images, default: none.')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=80013, type=int, metavar='N',
                    help='79077 for LightCNN-9/LightCNN-29, 80013 for LightCNN-29v2')
args = parser.parse_args()


################################################################################
def main():
    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=79077) #args.num_classes
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=79077) #args.num_classes
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=80013) #args.num_classes
    else: print('Error model type!\n')
    
    if not args.save_path: args.save_path = args.img_dir + "_ft"
    print(args)
    
    model.eval()
    if args.cuda==True: model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("\n==> Load checkpoint: %s" % args.resume)
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("\n==> NO checkpoint in: %s" % args.resume)

    transform = transforms.Compose([transforms.ToTensor()])
    input     = torch.zeros(1, 1, 128, 128)
    start     = time.time()
    for img_name in os.listdir(args.img_dir):
        img   = os.path.join(args.img_dir, img_name)
        img   = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img   = cv2.resize(img, (128, 128))
        img   = np.reshape(img, (128, 128, 1))
        img   = transform(img)
        input[0,:,:,:] = img

        if args.cuda==True: input = input.cuda()
        input_var   = torch.autograd.Variable(input)
        #input_var   = torch.autograd.Variable(input, volatile=True)
        _, feature = model(input_var)
        feat = feature.data.cpu().numpy()[0]
        save_feature(args.save_path, img_name, feat)
        print("\nFeature of %s:\n" % img_name, feature.data)
    print("\nTime = %f s\n" % (time.time()-start) )


def save_feature(save_path, img_name, feature):
    if not os.path.exists(save_path): os.makedirs(save_path)
    fname = os.path.join(save_path, img_name) + ".ft"
    #fname = os.path.splitext(fname)[0] + ".ft"
    with open(fname, 'wb') as ff: ff.write(feature)


################################################################################
if __name__ == '__main__':
    main()

