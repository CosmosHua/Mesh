'''
    Implement of training process for LightCNN
    @author: Alfred Xiang Wu @date: 2017.07.04
    @modifier: CosmosHua @date: 2019/01/10
'''

# coding:utf-8
# !/usr/bin/python3

import os, time
import numpy as np
import argparse, shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from load_imglist import ImageList


################################################################################
parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--model', default='', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29, LightCNN-29v2')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=80013, type=int,
                    metavar='N', help='number of classes (default: 80013)')
args = parser.parse_args()


################################################################################
def main():
    # create Light CNN for face recognition
    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else: print('Error model type\n')

    if args.cuda==True: model = torch.nn.DataParallel(model).cuda()
    print(args, "\n", model)

    # large lr for last fc parameters
    params = []
    for name, value in model.named_parameters():
        if 'bias' in name:
            if 'fc2' in name:
                params += [{'params':value, 'lr': 20 * args.lr, 'weight_decay': 0}]
            else:
                params += [{'params':value, 'lr': 2 * args.lr, 'weight_decay': 0}]
        else:
            if 'fc2' in name:
                params += [{'params':value, 'lr': 10 * args.lr}]
            else:
                params += [{'params':value, 'lr': 1 * args.lr}]

    cudnn.benchmark = True
    
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.cuda==True: criterion.cuda()
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.wd)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("\n==> Loading checkpoint: '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("==> Loaded:'{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("\n==> NO checkpoint in '{}'".format(args.resume))

    #load image
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path,
            fileList=args.train_list,
            transform=transforms.Compose([
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path,
            fileList=args.val_list,
            transform=transforms.Compose([
                transforms.CenterCrop(128),
                transforms.ToTensor(),])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #train model
    validate(val_loader, model, criterion)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch:
        train(train_loader, model, criterion, optimizer, epoch)
        
        # evaluate on validation set:
        prec1 = validate(val_loader, model, criterion)

        # save model checkpoint:
        save_name = args.save_path + 'lightCNN_' + str(epoch+1) + '.pth'
        model_state = {'epoch': epoch+1, 'arch': args.arch,
                      'state_dict': model.state_dict(), 'prec1': prec1}
        torch.save(model_state, save_name)


################################################################################
def adjust_learning_rate(optimizer, epoch):
    step  = 10
    scale = 0.457305051927326
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def train(train_loader, model, criterion, optimizer, epoch):
    end = time.time()
    model.train() # switch to training mode
    data_time, batch_time, losses, top1, top5 = [AverageMeter() for i in range(5)]
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input      = input.cuda()
        target     = target.cuda()
        input_var  = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, _ = model(input_var)
        loss      = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    end = time.time()
    model.eval() # switch to evaluate mode
    batch_time, losses, top1, top5 = [AverageMeter() for i in range(4)]
    for i, (input, target) in enumerate(val_loader):
        input      = input.cuda()
        target     = target.cuda()
        input_var  = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, _ = model(input_var)
        loss   = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
    print('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(losses.avg, top1.avg))
    return top1.avg


################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self): self.reset()

    def reset(self): self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


################################################################################
if __name__ == '__main__':
    main()