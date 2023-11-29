import argparse
import time
import datetime
import os, csv
import random
import shutil
import warnings
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.nn import functional as F

from models import create_model
from data import TrainM, TestM, Balance
from train import train, validate, Score, train2

parser = argparse.ArgumentParser(description='Age Estimate Training and Evaluating')

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.005)#
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=24)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--evaluation', type=bool, default=False)
parser.add_argument('--checkpoints', type=str, default="/bzh/GTA/Checkpoint/1of2_glink_23.pth")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N')
parser.add_argument('--seed', default=24, type=int)
parser.add_argument('--experiment', type=str, default='2of2_mae_')
parser.add_argument('--useonecycle', type=bool, default=True)
parser.add_argument('--stage', type=int, default=2)

def save_checkpoint(model, args, epoch):# 将最优的模型保存下来
    print('Best Model Saving...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch + 1
    }, os.path.join('checkpoints', args.experiment + str(epoch) + '.pth'))

def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # 设置GPU并行计算
    cudnn.deterministic = True
    cudnn.benchmark = False
    # 获得GPU数量
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank, args.nprocs, args)

def main_worker(local_rank, nprocs, args):
    torch.distributed.init_process_group(backend='nccl', init_method='env://')#引入环境变量

    model = create_model('efficientnet_v2s', num_classes=101)#, pretrained=True
    if args.stage == 2:
        model.freeze_features()#effecttnet.py冻结全连接层增加可复性

    if args.local_rank == 0:
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))#输出参数模型
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)# 归一化处理

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)##分配GPU

    args.batch_size = int(args.batch_size / nprocs)#分布式训练在多个GPU上训练，通过batchsize/进程数量（nprocs）
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = None#不设置损失函数

    if args.stage == 2:#('--lr', type=float, default=0.005)
        args.lr = 0.001
        args.epochs = 8

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    if args.checkpoints is not None:# 加载预训练
        checkpoints = torch.load(args.checkpoints, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoints['model_state_dict']) # 继续之前的训练，解耦学习
    
    transform = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_t = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=3),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),#train
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 训练transform_t
    if args.stage == 1:# default=2
        train_dataset = TrainM(transform_t)# 数据增强标签分布
    if args.stage == 2:
        train_dataset = Balance(transform_t)# 数据增强标签分布
    # 在分布式中对数据集进行采样sampler；然后创建一个train_loader，pin_memory 参数（如果为 True，则数据将被复制到 CUDA 固定内存中）
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=train_sampler)
    itern = len(train_loader)# 为dataloader中批次数目

    # 测试transform
    val_dataset = TestM(transform)
    # 在多个进程之间对验证集进行sample
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=val_sampler)

    if args.useonecycle is True:# default=True
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=itern, epochs=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)

    if args.evaluation:#defalut=Faluse
        val_metrics = validate(val_loader, model, local_rank, args)
        print(val_metrics['mae'])
        return
    
    if args.local_rank == 0:#local_rank=0
        writer = SummaryWriter(
            log_dir=f'runs/' + args.experiment
        )

    best_score = 7
    best_mae = 100

    for epoch in range(args.start_epoch, args.epochs):
        # 在训练和测试采样器加上当前的epoch
        # 根据stage的值选取第一阶段或者第二阶段的训练
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        if args.stage == 1:
            train_metrics = train(train_loader, model, optimizer, scheduler, criterion, epoch, local_rank, args)
        if args.stage == 2:
            mae = 1.72
            train_metrics = train2(train_loader, model, optimizer, scheduler, criterion, epoch, local_rank, mae, args)
        # 验证阶段
        val_metrics = validate(val_loader, model, local_rank, args)
            
        if args.local_rank == 0:#开启主程序
            writer.add_scalar('Train_loss', train_metrics['loss'], epoch + 1)# writer用于写入Tensorboard
            writer.add_scalar('Val_mae', val_metrics['mae'], epoch + 1)
            for param_group in optimizer.param_groups:# 遍历了优化器参数组。将学习率写入Tensorboard
                writer.add_scalar('Lr_rate', param_group['lr'], epoch + 1)

        if args.useonecycle is False:# default=True更新学习率
            scheduler.step()

        if args.stage == 1:
            is_best = val_metrics['mae'] < best_mae 
            best_mae = min(val_metrics['mae'], best_mae)
        
        if args.stage == 2:
            score = Score(model, local_rank, transform, val_metrics['mae'], args)
            is_best = score > best_score # score为AAR的值
            best_score = max(score, best_score)

        if args.local_rank == 0 and is_best:
            save_checkpoint(model, args, epoch)

    if args.local_rank == 0:#
        if args.stage == 1:
            print('Best Mae: {0}'.format(best_mae))
        else:
            print('Best Score: {0}'.format(best_score))

if __name__ == '__main__':
    main()
