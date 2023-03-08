#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 19 Sep, 2019

@author: wangshuo
"""

import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import metric
from utils import collate_fn
from PIGR import PIGR
from dataset import load_data, RecSysDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/gowalla/', help='dataset directory path: datasets/gowalla/yoochoose/lastfm')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=150, help='hidden state size of gru module')
parser.add_argument('--epoch', type=int, default=15, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--layers', type=int, default=1, help='learning layers')
parser.add_argument('--neighbors', type=int, default=1, help='neighbors')
parser.add_argument('--ab', type=bool, default=False, help='whether conduct ablation study')
parser.add_argument('--lr_dc', type=float, default=0.2, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=10, help='the number of steps after which the learning rate decay')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
def main():
    print('Loading data...')

    train, test, side_info = load_data(args.dataset_path, valid_portion=args.valid_portion)

    train_data = RecSysDataset(train)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = 4, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = 4, collate_fn = collate_fn)

    model = PIGR(args.hidden_size, args.batch_size, side_info, args.ab, device, args.layers, args.neighbors).to(device)

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=[args.lr_dc_step], gamma=0.1)

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        model.epoch = epoch
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 200)
        scheduler.step()
        best_result = [0,0]
        recall5, mrr5, recall10, mrr10, recall20, mrr20, recall50, mrr50, recall100, mrr100 = validate(test_loader, model, [5, 10, 20, 50, 100])
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, 5, recall5, 5, mrr5))
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, 10, recall10, 10, mrr10))
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, 20, recall20, 20, mrr20))
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, 50, recall50, 50, mrr50))
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, 100, recall100, 100, mrr100))
        # store best loss and save a model checkpoint
        if recall20 > best_result[0] and mrr20 > best_result[1]:
            best_result[0] = recall20
            best_result[1] = mrr20
            ckpt_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()
    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens, reverse_lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        lens = lens.to(device)
        reverse_lens = reverse_lens.to(device)
        optimizer.zero_grad()
        pre_scores, conloss = model(seq, lens, reverse_lens)
        celoss = criterion(pre_scores, target)
        loss = celoss + conloss
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()

        sum_epoch_loss += batch_loss

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                  % (epoch + 1, num_epochs, batch_loss, sum_epoch_loss / (i + 1),
                     len(seq) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model, metrics):
    model.eval()
    records = [[]]*len(metrics)*2
    with torch.no_grad():
        for seq, target, lens, reverse_lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            lens = lens.to(device)
            reverse_lens = reverse_lens.to(device)
            outputs, _ = model(seq, lens, reverse_lens)
            logits = F.softmax(outputs, dim=1)
            for i, one_metric in enumerate(metrics):
                recall, ndcg = metric.evaluate(logits, target, device, k=one_metric)
                records[2*i] = records[2*i] + recall
                records[2*i+1] = records[2*i+1] + ndcg
    res = []
    for record in records:
        res.append(np.mean(record))

    return tuple(res)



if __name__ == '__main__':
    main()
