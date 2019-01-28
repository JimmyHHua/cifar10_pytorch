#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-01-24 16:02:42

@author: JimmyHua
"""

import argparse
import logging
import sys

import torch
from torch.backends import cudnn
import network.dpnet
from core.loader import get_data_set
from core.train_engine import Train_Engine
#import torchvision.models as models

logging.basicConfig(
    level = logging.INFO, #打印日志级别数值
    format = '%(asctime)s: %(message)s', #输出时间和信息
    stream=sys.stdout #指定日志的输出流
    )

def train(args):
    train_data, valid_data, train_valid_data = get_data_set(args.bs)
    net = network.dpnet.DPN92()
    #net.load_state_dict(torch.load('checkpoints/model_best.pt')['state_dict'])
    #print('********model_best has been loaded********')
    print('********dpn92 has been loaded********')
    net.apply(network.dpnet.conv_init)
    optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,weight_decay=args.wd,momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    net = torch.nn.DataParallel(net)
    if args.use_gpu:
        net = net.cuda()
    model = Train_Engine(net,args.use_gpu)
    model.fit(train_data=train_data, test_data=valid_data, optimizer=optimizer, criterion=criterion,
            epochs=args.epochs, print_interval=args.print_interval, eval_step=args.eval_step,save_step=args.save_step, save_dir=args.save_dir)
def main():
    parser = argparse.ArgumentParser(description='cifar10 trainning')
    parser.add_argument('--bs',type=int, default=128, help='batch_size')
    parser.add_argument('--lr',type=float, default=0.005, help='learning rate')
    parser.add_argument('--wd',type=float, default=3e-4, help='weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--epochs', type=int, default=600, help='training epochs')
    parser.add_argument('--print_interval', type=int, default=10, help='how many iterations to print')
    parser.add_argument('--eval_step', type=int, default=1, help='how many epochs to evaluate')
    parser.add_argument('--save_step', type=int, default=50, help='how many epochs to save model')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='save model directory')
    parser.add_argument('--use_gpu', action='store_true', help='decide if use gpu training')

    args=parser.parse_args()
    cudnn.benchmark = True
    train(args)

if __name__ == '__main__':
    main()
