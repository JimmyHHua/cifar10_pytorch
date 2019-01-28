#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-01-24 18:07:47

@author: JimmyHua
"""
import argparse
import logging
import os
import sys

import pandas as pd
import torch
from torch import nn
from torch.backends import cudnn

import network.dpnet
from core.loader import get_test_set

logging.basicConfig(
    level = logging.INFO, #打印日志级别数值
    format = '%(asctime)s: %(message)s', #输出时间和信息
    stream=sys.stdout #指定日志的输出流
    )

def submit(args):
    test_loader, id_to_class = get_test_set(args.bs)
    net = network.dpnet.DPN92()
    net.load_state_dict(torch.load('checkpoints/200_checkpoint.pt')['state_dict'])
    net = nn.DataParallel(net)
    net.eval()
    if args.use_gpu:
        net = net.cuda()

    pred_labels = list()
    indices = list()
    print('model has been loaded!')
    for data, fname in test_loader:
        if args.use_gpu:
            data = data.cuda()
        with torch.no_grad():
            scores = net(data)
        labels = scores.max(1)[1].cpu().numpy()
        pred_labels.extend(labels)
        indices.extend(fname.numpy())
    df = pd.DataFrame({'id': indices, 'label': pred_labels})
    df['label'] = df['label'].apply(lambda x: id_to_class[x])
    df.to_csv('submission.csv', index=False)
    print('Finished!')


def main():
    parser = argparse.ArgumentParser(description='cifar10 model testing')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_best.pth.tar',
                        help='training batch size')
    parser.add_argument('--bs', type=int, default=128, help='testing batch size')
    parser.add_argument('--use_gpu', action='store_true', help='decide if use gpu training')

    args = parser.parse_args()
    cudnn.benchmark = True
    submit(args)


if __name__ == '__main__':
    main()