#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import datetime
import numpy as np

from model.coarse2fine import FinerRes2CSNet
from dataloader.trine import Data

from utils.train_metrics import metrics3d
from losses.diceloss import DiceLoss
from losses.weighted_cross_entropy import WeightedCrossEntropyLoss

"""
Classes:
    0: background
    1: tissue
    2: vascular
    3: trigeminal nerve
"""

args = {
    'root'      : '/',
    'data_path' : 'dataset/TRINE/',
    'epochs'    : 1000,
    'lr'        : 0.0001,
    'snapshot'  : 100,
    'test_step' : 1,
    'ckpt_path' : './checkpoint/',
    'batch_size': 8,
    'num_class' : 4
}


def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    date = datetime.datetime.now().strftime("%Y_%m_%d_")
    torch.save(net, args['ckpt_path'] + 'FinerRes2CSNet-' + iter + '.pth')
    print("{} Saved model to:{}".format("\u2714", args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    net = FinerRes2CSNet(in_channels=1, classes=args['num_class']).cuda()
    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)

    # load train dataset
    train_data = Data(args['data_path'], train=True)
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=4, shuffle=True)

    critrion1 = DiceLoss().cuda()
    critrion2 = WeightedCrossEntropyLoss().cuda()
    mse = nn.MSELoss().cuda()
    # Start training
    print("\033[1;30;44m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))

    iters = 1
    best_tp, best_dc = 0., 0.
    for epoch in range(args['epochs']):
        net.train()
        TPR, FNR, FPR, DSC, DSC_Nerve = [], [], [], [], []
        for idx, batch in enumerate(batchs_data):
            image = batch[0].cuda()
            label = batch[1].cuda()

            optimizer.zero_grad()
            coarse, fine = net(image)
            label = label.squeeze_(1)  # for CE Loss series
            loss1 = critrion1(coarse, label) + critrion2(coarse, label)
            loss2 = critrion1(fine, label) + critrion2(fine, label)
            # loss_wce = critrion4(pred, label) + critrion4(coarse, label)
            loss = (loss1 + loss2) / 4.0
            loss.backward()
            optimizer.step()

            pred = coarse
            tp, fn, fp, iou, dice = metrics3d(pred, label, pred.shape[0])
            TPR.append(np.mean(tp))
            FNR.append(np.mean(fn))
            FPR.append(np.mean(fp))
            DSC.append(np.mean(dice))
            DSC_Nerve.append(dice[3])
            if epoch % 2 == 0:
                print(
                    '\033[1;36m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tDSC:{6:.4f}\tnum_class:{7}\033[0m'.format(
                        epoch + 1, iters, loss.item(), np.mean(tp), np.mean(fn), np.mean(fp), np.mean(dice),
                        torch.unique(torch.argmax(pred, dim=1))))
            else:
                print(
                    '\033[1;32m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tDSC:{6:.4f}\tnum_class:{7}\033[0m'.format(
                        epoch + 1, iters, loss.item(), np.mean(tp), np.mean(fn), np.mean(fp), np.mean(dice),
                        torch.unique(torch.argmax(pred, dim=1))))

            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------

        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)

        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, str(epoch + 1))


if __name__ == '__main__':
    train()
