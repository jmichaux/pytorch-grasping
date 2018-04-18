import argparse
import copy
import random
from os.path import basename, dirname, join
import glob
import numbers
import numpy as np
import pandas as pd
import time

from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import bytescale

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, utils
import torchvision.transforms as t
from torchvision.transforms import functional as f

from datasets import *
from intersection import *
from transforms import *


parser = argparse.ArgumentParser(description="Pytorch grasp detection")
parser.add_argument('--data', dest='data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--csv', dest='csv_dir', metavar='CSV',
                    help='path to csv file')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100000, type=int, metavar='N',
                    help='number of total epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful for restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch-print-freq', '--bp', default=10, type=int,
                    metavar='N', help='print frequency for batches (default: 10)')
parser.add_argument('--epoch-print-freq', '--ep', default=1, type=int,
                    metavar='N', help='print frequency for batches (default: 1)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pretrained model')
parser.add_argument('--train-all', dest='train_all', action='store_false',
                    help='make entire network trainable')
parser.add_argument('--grasp-config', default=5, type=int,
                    help='parameterizaton length of grasps')
parser.add_argument('--num-folds', default=5, type=int,
                    help='number of cross-validation folds')
# parser.add_argument('--verbose', '-v', dest='verbose', action='store_true',
#                     help='print during training')


######################
#   Data Transforms  #
######################

# pre transforms
pre_img_transform = t.Compose([
    t.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
])

pre_pcd_transform = t.Compose([
    PCDtoRGB(),
])

# post transforms
post_img_transform = t.Compose([
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

post_pcd_transform = t.Compose([
    t.ToTensor(),
    t.Normalize(mean=[0.406, 0.406, 0.406], std=[0.225, 0.225, 0.225])
])

target_transform = t.Compose([
    TargetTensor(),
])

# co_transforms
train_co_transform = Compose([
    RandomRotation(40),
    RandomTranslation(50),
    CenterCrop(320),
    Resize(224),
    RandomVerticalFlip(),
    RandomHorizontalFlip(),
])

# validation co-transorms
val_co_transform = Compose([
    CenterCrop(320),
    Resize(224),
])

######################
#      Functions     #
######################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(prediction, target):
    # Check that the orientation of the target and
    # prediction boxes differ by less than 30 degrees
    thetas = (np.abs(np.array(prediction[:, -1]) - np.array(target[:, -1])) < np.radians(30)).astype(int)

    # compute intersection over union
    pred_rect = np.array(prediction).tolist()
    target_rect = np.array(target).tolist()
    ious = []
    for pred, tar in zip(pred_rect, target_rect):
        intersection = intersection_area(pred, tar)
        union = pred[2] * pred[3] + tar[2] * tar[3] - intersection
        ious.append(intersection / union)
    ious = (np.array(ious) > 0.25).astype(int)
    return np.mean(thetas * ious)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target, _, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # convert inputs and targets to torch Variables
        input = input.cuda()
        target = target.float().cuda()

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        acc.update(accuracy(output.data, target.data), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()    
    return losses.avg, acc.avg



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    for i, (input, target, _, _) in enumerate(val_loader):
        input = input.cuda()
        target = target.float().cuda()

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        # compute ouput 
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        acc.update(accuracy(output.data, target.data), input.size(0))

    return losses.avg, acc.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():

    args = parser.parse_args()
    fold = 0
    # for fold in range(args.num_folds):
    # create dataset

    train_data = CornellGraspingDataset(
    csv_file=args.csv_dir, 
    data_dir=args.data_dir,
    fold=fold,
    split='train',
    split_type='image',
    use_pcd=False,
    concat_pcd=False,
    pre_img_transform=None,
    pre_pcd_transform=None,
    co_transform=train_co_transform,
    post_img_transform=post_img_transform,
    post_pcd_transform=None,
    target_transform=target_transform,
    grasp_config=args.grasp_config,)


    train_loader = DataLoader(train_data,
                      batch_size=32,
                      shuffle=True,
                      num_workers=4,
                      pin_memory=True)


    val_data = CornellGraspingDataset(
        csv_file=args.csv_dir, 
        data_dir=args.data_dir,
        fold=fold,
        split='val',
        split_type='image',
        use_pcd=False,
        concat_pcd=False,
        pre_img_transform=None,
        pre_pcd_transform=None,
        co_transform=val_co_transform,
        post_img_transform=post_img_transform,
        post_pcd_transform=None,
        target_transform=target_transform,
        grasp_config=args.grasp_config,)


    val_loader = DataLoader(val_data,
                      batch_size=32,
                      shuffle=False,
                      num_workers=4,
                      pin_memory=True)

    #Load model
    model = torchvision.models.alexnet(pretrained=True)

    # If only training the last two layers
    if not args.train_all:
        for param in model.parameters():
            param.requires_grad = False

    # Replace last two linear layers
    model.classifier._modules['4'] = nn.Linear(4096, 512)
    model.classifier._modules['6'] = nn.Linear(512, args.grasp_config)
    model.cuda()

    # Loss, optimizer, lr_scheduler
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.1)

    epoch_time = AverageMeter()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()        
        # one training step
        t_loss, t_acc = train(train_loader, model, criterion, optimizer, epoch)
        
        # one validation step
        v_loss, v_acc = validate(val_loader, model, criterion)

        # update loss, accuracy, and epoch time
        epoch_time.update(time.time() - end)
        train_loss.update(t_loss, 1)
        train_acc.update(t_acc, 1)
        val_loss.update(v_loss, 1)
        val_acc.update(v_acc, 1)

        print('Epoch: {0}/{1}\t'
              'Time {time.val:.3f} ({time.avg:.3f})\t'
              'Train Loss {t_loss.val:.4f} ({t_loss.avg:.4f})\t'
              'Train Acc {t_acc.val:.3f} ({t_acc.avg:.3f})\t'
              'Val Loss {v_loss.val:.4f} ({v_loss.avg:.4f})\t'
              'Val Acc {v_acc.val:.3f} ({v_acc.avg:.3f})'.format(
                                                                 epoch+1, args.epochs-args.start_epoch, time=epoch_time,
                                                                 t_loss=train_loss, t_acc=train_acc,
                                                                 v_loss=val_loss, v_acc=val_acc))

    print("Training time: {}".format(time.time() - start))


if __name__ == '__main__':
    main()
