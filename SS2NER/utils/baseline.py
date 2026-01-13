#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : W2NER_multiTask.py
@Author  : HuYing
@Time    : 2022/12/26 22:04
@Description: 
"""
import os
import time

import numpy as np
import torch
import random
import json
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def call_f1(c, p, r):
    """
    :param c: 预测为T且标签为T的个数
    :param p: 预测为T的个数
    :param r: 数据集中T的个数
    :return:
    """
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return p, r, 2 * p * r / (p + r)
    return p, r, 0


def decode(outputs, targets, entity_text, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    for sample_i in range(len(outputs)):  # one sentence
        predicts = []
        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i].to(targets[sample_i].device)
        for e_i in range(len(output)):
            pred_boxes = output[e_i][:4]
            pred_scores = output[e_i][4]  # 置信度
            pred_conf_score = output[e_i][5]  # 类别概率
            pred_labels = output[e_i][-1]
            if int(pred_labels.cpu().numpy()) <= 0:
                continue
            start = int(np.around(pred_boxes[0].cpu().numpy()))
            end = int(np.around(pred_boxes[-1].cpu().numpy()))
            label = int(pred_labels.cpu().numpy())

            if (start >= 0) and (end > 0) and (start < end) and (start < length[sample_i]) and (end <= length[sample_i]):
                # if (pred_scores < 0.95) or (pred_conf_score < 0.7): continue
                predicts.append(([ind for ind in range(start, end)], label))

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        try:
            ent_r += len(entity_text[sample_i])
            ent_p += len(predicts)
            ent_c += len(predicts.intersection(entity_text[sample_i]))
        except:
            print(entity_text, sample_i)

    return ent_c, ent_p, ent_r, decode_entities


def decode_search(outputs, entity_text, length):
    best_p, best_r, best_f = 0.0, 0.0, 0.0
    for s in np.arange(0.45, 0.99, 0.01):
        for c_s in np.arange(0.4, 0.99, 0.05):
            total_ent_r = 0
            total_ent_p = 0
            total_ent_c = 0
            ent_r, ent_p, ent_c = 0, 0, 0
            decode_entities = []
            for sample_i in range(len(outputs)):  # one sentence
                predicts = []
                if outputs[sample_i] is None:
                    continue

                output = outputs[sample_i]
                for e_i in range(len(output)):
                    pred_boxes = output[e_i][:4]
                    pred_scores = output[e_i][4]  # 置信度
                    pred_conf_score = output[e_i][5]  # 类别概率
                    pred_labels = output[e_i][-1]
                    if int(pred_labels.cpu().numpy()) <= 0:
                        continue
                    start = int(np.around(pred_boxes[0].cpu().numpy()))
                    end = int(np.around(pred_boxes[-1].cpu().numpy()))
                    label = int(pred_labels.cpu().numpy())
                    if (start >= 0) and (end > 0) and (start < end) and (start < length[sample_i]) and (end <= length[sample_i]):
                        if (pred_scores < s) or (pred_conf_score < c_s): continue
                        predicts.append(([ind for ind in range(start, end)], label))

                predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
                decode_entities.append([convert_text_to_index(x) for x in predicts])
                ent_r += len(entity_text[sample_i])
                ent_p += len(predicts)
                ent_c += len(predicts.intersection(entity_text[sample_i]))

            def call_f1(c, p, r):
                if r == 0 or p == 0:
                    return 0, 0, 0

                r = c / r if r else 0
                p = c / p if p else 0

                if r and p:
                    return p, r, 2 * p * r / (p + r)
                return p, r, 0

            total_ent_r += ent_r
            total_ent_p += ent_p
            total_ent_c += ent_c
            rel_p, rel_r, rel_f = call_f1(total_ent_c, total_ent_p, total_ent_r)
            if rel_f > best_f:
                best_p = rel_p
                best_r = rel_r
                best_f = rel_f
                print("current PRF: %.4f, %.4f, %.4f" % (best_p, best_r, best_f), "current score, conf_score: ", s, c_s)

    return best_p, best_r, best_f


def plot_grid_with_rectangles(n, rectangles, config):
    # 创建一个 n × n 的网格
    grid = np.zeros((n, n))

    # 创建一个新的图形
    fig, ax = plt.subplots()

    # 绘制网格线
    ax.grid(True, which='both', color='black', linewidth=1)

    # 设置刻度范围
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    # 颜色映射字典
    color_map = {1: 'red', 2: 'blue'}

    # 遍历矩形列表并绘制矩形
    for rect in rectangles:
        x_start, y_start, x_end, y_end, color, label = rect
        label = str(str(x_start) + "-" + str(y_end) + "-" + label)
        width = x_end - x_start
        height = y_end - y_start
        rectangle = plt.Rectangle((x_start, y_start), width, height, fill=False, fc=color_map[color])
        ax.add_patch(rectangle)
        ax.text(x_start, y_end, label, fontsize=12, verticalalignment='top')

    # 显示图形
    path = config.output_dir + "/figures"
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(os.path.join(path, str(len(os.listdir(path))) + ".jpg"))


def show_results(outputs, entity_text, length, config):
    for sample_i in range(len(outputs)):  # one sentence
        predicts = []
        ground_truth = []
        if outputs[sample_i] is None:
            continue

        for entity in entity_text[sample_i]:
            ground_truth.append((int(entity.split("#")[0].split("-")[0]), int(entity.split("#")[0].split("-")[-2]) + 1, entity.split("#")[1][1:]))

        output = outputs[sample_i]
        for e_i in range(len(output)):
            pred_boxes = output[e_i][:4]
            pred_labels = output[e_i][-1]
            if int(pred_labels.cpu().numpy()) <= 0:
                continue
            start = int(np.around(pred_boxes[0].cpu().numpy()))
            end = int(np.around(pred_boxes[-1].cpu().numpy()))
            label = int(pred_labels.cpu().numpy())
            if (start >= 0) and (end > 0) and (start < end) and (start < length[sample_i]) and (end < length[sample_i]):
                predicts.append((start, end, str(label)))

        # 颜色映射字典
        if set(predicts) != set(ground_truth):
            pred_rectangles = [(i, i, j, j, 1, str(k)) for i, j, k in predicts]
            true_rectangles = [(i, i, j, j, 2, str(k)) for i, j, k in ground_truth]
            plot_grid_with_rectangles(length[sample_i], list(set(pred_rectangles) ^ set(true_rectangles)), config)


def load_embeddings(embedding_file):
    vocab2id = {}
    embedding = []

    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            vector = list(map(float, line[1:]))
            word = line[0]
            vocab2id[word] = len(vocab2id) + 2
            embedding.append(vector)

        vocab2id["<UNK>"] = 1
        vector = np.random.randn(len(vector))
        embedding.append(vector)

        vocab2id["<PAD>"] = 0
        vector = np.random.randn(len(vector))
        embedding.append(vector)
    return vocab2id, np.array(embedding, dtype=np.float32)


def load_char_pos_vocab(path):
    with open(path, encoding='utf-8') as file:
        char_pos_dict = json.load(file)
    char2id = {i: j + 2 for j, i in enumerate(char_pos_dict["char"])}
    pos2id = {i: j + 2 for j, i in enumerate(char_pos_dict["pos"])}
    char2id["<UNK>"] = 1
    char2id["<PAD>"] = 0
    char2id["<END>"] = len(char2id)
    pos2id["<UNK>"] = 1
    pos2id["<PAD>"] = 0
    pos2id["<END>"] = len(pos2id)

    return char2id, pos2id


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def bbox_iou(box1, box2, conf_scores=None, x1y1x2y2=True, sigma=0.5):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = (inter_area / (b1_area + b2_area - inter_area + 1e-16))
    # a = -iou * iou / sigma
    # scores = conf_scores * torch.exp(-iou * iou / sigma)
    # print(scores)
    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # batch_size, anchors, 11(x1,y1,x2,y2,conf,类别概率)
    output = [None for _ in range(len(prediction))]
    t = time.time()
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        # detections: x1,y1,x2,y2, 置信度, 该类别的概率, 类别
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4], conf_scores=detections[:, 4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]  # 判断类别
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match  # iou大于阈值且类型相同
            weights = detections[invalid, 4:5]  # 置信度
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]  # 删除框
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

        if (time.time() - t) > 10:
            break

    return output


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def main():
    print('this message is from main function')


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
