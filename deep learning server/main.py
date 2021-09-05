import datetime
import argparse
import os
import re
import random
import shutil
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import transforms

from data.data_reader import DatasetReader
from data.data_splitter import DatasetSplit
from data.data_transformer import DatasetTransform
from data.transforms import SelectFrames, FrameDifference, Downsample, TileVideo, RandomCrop, Resize, RandomHorizontalFlip, Normalize, ToTensor

model_names = ['E', 'E_bi', 'E_bi_avg_pool', 'E_bi_max_pool']
data_names = ['FD','RWF','AH','UCF','ALL', 'YT']


parser = argparse.ArgumentParser(description='PyTorch Violence Predictor Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('data_name', default='H',
                    help='data name: ' +
                    ' | '.join(data_names) +
                    ' (default: H)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='E_bi_max_pool',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: E)')
parser.add_argument('--t', '--transform', dest='transform', default='RC', type=str,
                    help='type of data transform [R: resize, C: random crop and flip, RC: both] (default: R)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-1, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--c', '--cpu', dest='cpu', action='store_true',
                    help='evaluate model on cpu')
parser.add_argument('--k', '--kfold', dest='kfold', default=0, type=int,
                    help='evaulate model with kfold index as test')
parser.add_argument('--s', '--split', dest='split', default=4, type=int,
                    help='fractional split of training/validation data. I.e. 5 -> 1/5 data is validation')
parser.add_argument('--f', '--frames', dest='frames', default=20, type=int,
                    help='number of frame diffs per video')
parser.add_argument('--i', '--id', dest='ID', default='', type=str,
                    help='id of the learning')
best_prec = 0

seed = 250


def main():
    global args, best_prec
    args = parser.parse_args()

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = DatasetReader(root_dir=args.data, data_name=args.data_name)

    assert args.kfold < args.split, "kfold index must be less than the split size!"
    folds = fold(args.split, data)
    for i in range(args.kfold):
        next(folds)

    (train_dataset, val_dataset) = next(folds)

    def transform_factory(transformation):
        if(transformation == "RC"):
            train_transform = transforms.Compose([Resize(size=256), RandomCrop(size=224), RandomHorizontalFlip()])
        elif(transformation == "R"):
            train_transform = transforms.Compose([Resize(size=224)])
        elif(transformation == "RF"):
            train_transform = transforms.Compose([Resize(size=224), RandomHorizontalFlip()])

        return train_transform
    
    print('dataset loading')

    train_transformations = transforms.Compose([transform_factory(args.transform), SelectFrames(num_frames=args.frames), FrameDifference(dim=0), Normalize(), ToTensor()])
    val_transformations = transforms.Compose([Resize(size=224), SelectFrames(num_frames=args.frames), FrameDifference(dim=0), Normalize(), ToTensor()])
    
    train_dataset = DatasetTransform(train_dataset, train_transformations)
    val_dataset = DatasetTransform(val_dataset, val_transformations)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    print('dataset loaded')

    # create model
    print("=> creating model '{}'".format(args.arch))
    VP = network_factory(args.arch)

    model = VP()

    if not args.cpu:
        model = model.cuda()
    print('model loaded')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)

    loss = list()
    acc = list()
    prec = list()
    graph_path = '/content/gdrive/Shareddrives/2021청년인재_고려대과정_10조/BiConvLSTM_Violence_Detection_Spatiotemporal_Encoder/acc_graph/'
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            f_name = os.path.basename(args.resume)
            graph = np.load(graph_path+'석기acc_loss_prec_' + re.findall(r'_t(.+).tar',f_name)[0]+'.npy', allow_pickle=True)
            print('loading model saved on ', re.findall(r'_t(.+).tar',f_name)[0])

            acc = graph[0].tolist()
            loss = graph[1].tolist()
            prec = graph[2].tolist()

            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            print('start_epoch : ', args.start_epoch)
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model)
        return

    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    run_training(train_loader, val_loader, model, criterion, optimizer, best_prec, nowDatetime, acc, loss, prec, graph_path, args.ID)


def run_training(train_loader, val_loader, model, criterion, optimizer, best_prec, starttime, accuray_list,loss_list, precision_list, graph_path, ID):

    epoch = 0
    print('run training start')
    epoch_start = time.time()
    while True:
        # train for one epoch
        acc, loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec = validate(val_loader, model)
        
        # acc, loss, prec 저장
        accuray_list.append(acc)
        loss_list.append(loss)
        precision_list.append(prec)

        # graph 및 acc,loss,prec 저장
        save_accuracy_graph(graph_path, accuray_list, loss_list, precision_list, starttime)
        np.save(graph_path + '석기acc_loss_prec_' + starttime ,[accuray_list,loss_list,precision_list])

        # remember best prec@1 and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
            }, is_best, args.arch + "_" + args.data_name + "_fold_" + str(args.kfold), starttime=starttime)#d=ID,
        
        epoch += 1
        print('epcoch took : ', round(time.time()-epoch_start,2), ' sec')

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not args.cpu:
            target = target.cuda(non_blocking=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        if not args.cpu:
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        
        # compute output
        output_dict = model(input_var)
        loss = criterion(output_dict['classification'], target_var)

        # measure accuracy and record loss
        precision = accuracy(output_dict['classification'], target)
        losses.update(loss.data, input.size(0))
        prec.update(precision, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Data {data_time.avg:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Accuracy {prec.avg:.4f}'.format(
              epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
              data_time=data_time, loss=losses, prec=prec))
    return prec.avg, losses.avg

def validate(val_loader, model):
    batch_time = AverageMeter()
    prec = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        if not args.cpu:
            target = target.cuda(non_blocking=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        if not args.cpu:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output_dict = model(input_var)

        # measure accuracy and record loss
        # print('target : ',target)
        # print('----------\n accuracy : ', accuracy(output_dict['classification'], target))    
        precision = accuracy(output_dict['classification'], target)
        prec.update(precision, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: [{0}/{1}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Accuracy {prec.avg:.4f}'.format(
              i + 1, len(val_loader), batch_time=batch_time, prec=prec))

    return prec.avg


def save_checkpoint(state, is_best,id='someid', starttime='tmp'): #id='someid'
    save_model_path = '/content/gdrive/Shareddrives/2021청년인재_고려대과정_10조/Server/model/'
    filename = save_model_path + 'checkpoint.' + str(id) +'_t'+ starttime + '.tar' #str(id) +
    torch.save(state, filename)
    if is_best:
        model_best_filename = save_model_path + 'model_best.' +str(id) +'_t' +starttime+'.tar' # str(id) + 
        shutil.copyfile(filename, model_best_filename)
        
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


def accuracy(output, target):
    """Computes the precision of model"""
    batch_size = target.size(0)

    _, pred = torch.max(output.data, 1)
    correct = pred.eq(target).cpu().sum()

    res = correct * 100.0 / batch_size
    return res

def fold(folds, data): 
    tot_length = len(data) 
    split_length = tot_length // folds
    for i in range(folds): 
        train_dataset = DatasetSplit(data, (i + 1) * split_length, tot_length - split_length) 
        val_dataset = DatasetSplit(data, i * split_length, split_length)
        yield (train_dataset, val_dataset)

def network_factory(arch):

    if arch == 'E':
        from networks.E import VP
    elif arch == 'E_bi_max_pool':
        from networks.E_bi_max_pool import VP
    else:
        assert 0, "Bad network arch name: " + arch

    return VP

def save_accuracy_graph(path, accuracy, loss, prec, starttime, filename='석기'): 
    # path에 filename + 시작시간 + '_acc_loss.jpg'로 저장할 예정
    plt.figure(figsize=(5,15))

    plt.subplot(3,1,1)
    plt.plot(accuracy,label='acc', color='r')
    plt.legend()
    plt.xlabel('epoch')

    plt.subplot(3,1,2)
    plt.plot(loss,label='loss', color='g')
    plt.legend()
    plt.xlabel('epoch')

    plt.subplot(3,1,3)
    plt.plot(prec,label='precision', color='b')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(path+filename+starttime+'_acc_loss.jpg')
    #plt.show()
  
if __name__ == '__main__':
    main()
