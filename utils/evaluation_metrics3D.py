#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File   : evaluation_metrics3D.py
import numpy as np
from sklearn import metrics


def numeric_score(pred, gt):
    y_true = gt
    y_pred = pred
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    cm = metrics.confusion_matrix(y_true, y_pred, labels=np.arange(4))
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return FP, FN, TP, TN


def Dice(pred, gt):
    FP, FN, TP, TN = numeric_score(pred, gt)
    DSC = 2 * TP / (2 * TP + FP + FN)
    return DSC


def IoU(pred, gt):
    pred = np.int64(pred // 255)
    gt = np.int64(gt // 255)
    m1 = np.sum(pred[gt == 1])
    m2 = np.sum(pred == 1) + np.sum(gt == 1) - m1
    iou = m1 / m2
    return iou


def metrics_3d(pred, gt):
    FP, FN, TP, TN = numeric_score(pred, gt)
    tpr = TP / (TP + FN + 1e-10)
    fnr = FN / (FN + TP + 1e-10)
    fpr = FN / (FP + TN + 1e-10)
    iou = TP / (TP + FN + FP + 1e-10)
    dice = 2. * TP / (2. * TP + FP + FN + 1e-10)
    return tpr, fnr, fpr, iou, dice


def over_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    OR = Os / (Rs + Os)
    return OR


def under_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Us = np.float(np.sum((pred == 0) & (gt == 255)))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    UR = Us / (Rs + Os)
    return UR
