import numpy as np
import os
import torch.nn as nn
import torch

from utils.evaluation_metrics3D import metrics_3d


def threshold(image):
    # t = filters.threshold_otsu(image, nbins=256)
    image[image >= 127] = 255
    image[image < 127] = 0
    return image


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 255) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 255)))
    TP = np.float(np.sum((pred == 255) & (gt == 255)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def metrics(pred, label, batch_size):
    # pred = torch.argmax(pred, dim=1) # for CE Loss series
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    Acc, SEn = 0., 0.
    for i in range(batch_size):
        img = outputs[i, :, :]
        gt = labels[i, :, :]
        acc, sen = get_acc(img, gt)
        Acc += acc
        SEn += sen
    return Acc, SEn


def metrics3dmse(pred, label, batch_size):
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    tp, fn, fp, IoU = 0, 0, 0, 0
    for i in range(batch_size):
        img = outputs[i, :, :, :]
        gt = labels[i, :, :, :]
        tpr, fnr, fpr, iou = metrics_3d(img, gt)
        # dcr = Dice(img, gt)
        tp += tpr
        fn += fnr
        fp += fpr
        IoU += iou
    return tp, fn, fp, IoU


def metrics3d(pred, label, batch_size):
    pred = torch.argmax(pred, dim=1)  # for CE loss series
    outputs = pred.data.cpu().numpy().astype(np.uint8)
    labels = label.data.cpu().numpy().astype(np.uint8)

    tpr, fnr, fpr, iou, dice = metrics_3d(outputs, labels)
    return tpr, fnr, fpr, iou, dice


def get_acc(image, label):
    image = threshold(image)

    FP, FN, TP, TN = numeric_score(image, label)
    acc = (TP + TN) / (TP + FN + TN + FP + 1e-10)
    sen = (TP) / (TP + FN + 1e-10)
    return acc, sen
