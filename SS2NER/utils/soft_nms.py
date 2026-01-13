#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : soft_nms.py
@Author  : HuYing
@Time    : 2023/2/16 9:39
@Description: 
"""
import torch
import numpy as np


def bbox_iou(anchors1, anchors2):
    lt = np.minimum(anchors1[:, np.newaxis, :2], anchors2[:, :2])
    rb = np.maximum(anchors2[:, np.newaxis, 2:], anchors2[:, 2:])
    inter_area = np.prod(rb - lt, axis=1) * np.all(rb > lt, axis=1)
    area1 = np.prod(anchors1[2:] - anchors1[:2], axis=1)
    area2 = np.prod(anchors2[2:] - anchors2[:2], axis=1)
    return inter_area / (area1[:, np.newaxis] + area2 - inter_area)


def hard_nms(anchors, scores, iou_thresh=0.7, condidates_num=200):
    """
    Params:
        anchors(numpy.array): detection anchors before nms, with shape(N, 4)
        scores(numpy.array): anchor scores before nms, with shape(N, )
        iou_thresh(float): iou thershold

    Return:
        keeps(nump.array): keeped anchor indexes
    """
    # if no anchor in anchors
    if anchors.size == 0:
        return

        # sort by scores
    idxs = scores.argsort(0)[::-1]
    keeps = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        # if keeped num equal with condidates_num or only one anchor left
        if len(idxs) == 1 or len(keeps) == condidates_num:
            break

        idxs = idxs[1:]
        ious = bbox_iou(anchors[current][np.newaxis, :], anchors[idxs]).flaten()
        mask = ious <= iou_thresh

        idxs = idxs[mask]

    return np.array(keeps)


def soft_nms(anchors, scores, iou_thresh=0.7, condidates_num=200, sigma=0.5):
    """
    Params:
        anchors(numpy.array): detection anchors before nms, with shape(N, 4)
        scores(numpy.array): anchor scores before nms, with shape(N, )
        iou_thresh(float): iou thershold
        sigma(float): soft nms hyper-parameter

    Return:
        keeps(nump.array): keeped anchor indexes
    """
    # if no anchor in anchors
    if anchors.size == 0:
        return

    keeps = []
    while len(scores) > 0:
        # get the maximum score index
        max_idx = np.argmax(scores)

        keeps.append(max_idx)

        # if keeped num equal with condidates_num or only one anchor left
        if len(scores) == 1 or len(keeps) == condidates_num:
            break
        # get the left score indexes
        mask = np.arange(len(scores)) != max_idx
        scores = scores[mask]

        # calculate iou
        ious = bbox_iou(anchors[max_idx][np.newaxis, :], anchors[mask]).flaten()

        # re-asign value, scores decay
        scores = scores * np.exp(-ious * ious / sigma)
        scores = scores[scores > iou_thresh]
    return np.array(keeps)


if __name__ == '__main__':
    dets = np.array([[100, 120, 170, 200],
                     [20, 40, 80, 90],
                     [20, 38, 82, 88],
                     [200, 380, 282, 488],
                     [19, 38, 75, 91]])
    scores = np.array([
        [0.98],[0.99],[0.96],[0.9],[0.8]
    ])

    keep = soft_nms(dets, scores)
    print(keep)
