#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Combined.py
# @Time      :2023/5/13 17:04
# @Author    :Ying Hu
# @Desc      :
import torch

from model.layers import *
from transformers import AutoModel
import torch.nn.functional as F
# from model.CNN_Nested import JInter, JReduce, JReduce_recover
from model.W2NER import W2NER
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
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
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    nB = pred_boxes.size(0)  # batchsieze 4
    nA = pred_boxes.size(1)  # 每个格子对应了多少个anchor
    nC = pred_cls.size(-1)  # 类别的数量
    nG = pred_boxes.size(2)  # gridsize

    # Output tensors
    obj_mask = torch.zeros((nB, nA, nG, nG), dtype=torch.bool).to(pred_boxes.device)  # obj，anchor包含物体, 即为1，默认为0 考虑前景
    noobj_mask = torch.ones((nB, nA, nG, nG), dtype=torch.bool).to(pred_boxes.device)  # noobj, anchor不包含物体, 则为1，默认为1 考虑背景
    class_mask = torch.zeros((nB, nA, nG, nG), dtype=torch.float).to(pred_boxes.device)  # 类别掩膜，类别预测正确即为1，默认全为0
    iou_scores = torch.zeros((nB, nA, nG, nG), dtype=torch.float).to(pred_boxes.device)  # 预测框与真实框的iou得分
    tx = torch.zeros((nB, nA, nG, nG), dtype=torch.float).to(pred_boxes.device)  # 真实框相对于网格的位置
    ty = torch.zeros((nB, nA, nG, nG), dtype=torch.float).to(pred_boxes.device)
    tw = torch.zeros((nB, nA, nG, nG), dtype=torch.float).to(pred_boxes.device)
    th = torch.zeros((nB, nA, nG, nG), dtype=torch.float).to(pred_boxes.device)
    tcls = torch.zeros((nB, nA, nG, nG, nC), dtype=torch.float).to(pred_boxes.device)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG  # target中的xywh都是0-1的，可以得到其在当前gridsize上的xywh
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # 每一种规格的anchor跟每个标签上的框的IOU得分
    best_ious, best_n = ious.max(0)  # 得到其最高分以及哪种规格框和当前目标最相似
    # Separate target values
    b, target_labels = target[:, :2].long().t()  # 真实框所对应的batch，以及每个框所代表的实际类别
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()  # 位置信息，向下取整了
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1  # 实际包含物体的设置成1
    noobj_mask[b, best_n, gj, gi] = 0  # 相反

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):  # IOU超过了指定的阈值就相当于有物体了
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()  # 根据真实框所在位置，得到其相当于网络的位置
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1  # 将真实框的标签转换为one-hot编码形式
    # Compute label correctness and iou at best anchor 计算预测的和真实一样的索引
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)  # 与真实框想匹配的预测框之间的iou值

    tconf = obj_mask.float()  # 真实框的置信度，也就是1
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0.0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        out = o1 + o2
        out = out.permute(0, 3, 1, 2)
        return out


class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers
        self.use_word_embedding = config.use_word_embedding
        self.use_char_embedding = config.use_char_embedding
        self.bert = AutoModel.from_pretrained(config.model_name_or_path, output_hidden_states=True)
        self.bert_dropout = nn.Dropout(config.bert_dropout)
        self.emb_dropout = nn.Dropout(config.emb_dropout)
        if self.use_word_embedding:
            self.word_embedding_dim = config.word_embedding_dim
            self.word_embedding = nn.Embedding(len(config.word2id), self.word_embedding_dim)
            self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_word_embed))
        if self.use_char_embedding:
            self.char_embedding_dim = config.char_embedding_dim
            self.char_embedding = nn.Embedding(len(config.char2id), self.char_embedding_dim)
            self.char_lstm = nn.LSTM(input_size=self.char_embedding_dim, hidden_size=self.char_embedding_dim, num_layers=1, bidirectional=True,
                                     batch_first=True)

    @staticmethod
    def combine(sub, sup_mask, pool_type="max"):
        sup = None
        if len(sub.shape) == len(sup_mask.shape):
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup == -1e30] = 0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup == -1e30] = 0
        return sup

    def forward(self, input_ids, pieces2word, word_ids=None, char_count=None, char_ids=None, token_masks_char=None):
        bert_emb = self.bert(input_ids=input_ids, attention_mask=input_ids.ne(0).float())
        cls_embedding = bert_emb.last_hidden_state[:, 0]
        if self.use_bert_last_4_layers:
            bert_emb = torch.stack(bert_emb[2][-4:], dim=-1).mean(-1)
        else:
            bert_emb = bert_emb[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_emb).item()

        _bert_embs = bert_emb.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.bert_dropout(word_reps)
        batch_size = input_ids.size(0)

        embeds = [word_reps]
        if self.use_word_embedding:
            word_embed = self.word_embedding(word_ids)
            word_embed = self.emb_dropout(word_embed)
            embeds.append(word_embed)  # 词Embedding
        if self.use_char_embedding:
            char_count = char_count.view(-1)
            token_masks_char = token_masks_char
            max_token_count = char_ids.size(1)
            max_char_count = char_ids.size(2)
            char_encoding = char_ids.view(max_token_count * batch_size, max_char_count)
            # char_encoding[char_count == 0][:, 0] = 101
            char_count[char_count == 0] = 1
            char_embed = self.char_embedding(char_encoding)
            char_embed = self.emb_dropout(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(input=char_embed, lengths=char_count.tolist(), enforce_sorted=False, batch_first=True)
            char_embed_packed_o, (_, _) = self.char_lstm(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(batch_size, max_token_count, max_char_count, self.char_embedding_dim * 2)
            h_token_char = self.combine(char_embed, token_masks_char, "max")
            h_token_char = self.emb_dropout(h_token_char)
            embeds.append(h_token_char)

        embeds = torch.cat(embeds, dim=-1)
        return embeds, cls_embedding


class CNN_Nested(nn.Module):
    def __init__(self, config):
        super(CNN_Nested, self).__init__()
        hidden_size = config.bert_config.hidden_size
        size_embed_dim = 25
        biaffine_size = 200
        cnn_dim = 200
        n_head = 5
        kernel_size = 3
        cnn_depth = 3
        self.w2ner_out = config.w2ner_out
        if size_embed_dim != 0:
            n_pos = 30
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size * 2 + size_embed_dim + 2
        else:
            hsz = biaffine_size * 2 + 2
        biaffine_input_size = hidden_size
        if config.use_word_embedding:
            biaffine_input_size += config.word_embedding_dim
        if config.use_char_embedding:
            biaffine_input_size += config.char_embedding_dim * 2  # bi_lstm

        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.cls_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(config.bert_config.hidden_size, biaffine_size),
            nn.LeakyReLU(),
            nn.Linear(biaffine_size, 1),
        )

        self.dropout = nn.Dropout(0.4)
        if n_head > 0:
            self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth > 0:
            self.cnn = MaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=2)

        self.conv_l_pre_down = PMaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=1)
        self.conv_l_post_down = PMaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=1)

        self.conv_s_pre_down = PMaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=1)
        self.conv_s_post_down = PMaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=1)

        self.down_fc = nn.Linear(cnn_dim, self.w2ner_out)

    def forward(self, word_reps, cls_embeding, pieces_index, loss_mask):
        lengths, _ = pieces_index.max(dim=-1)

        head_state = self.head_mlp(word_reps)
        tail_state = self.tail_mlp(word_reps)
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.multi_head_biaffine(head_state, tail_state)

        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)

        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:word_reps.size(1), :word_reps.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(word_reps.size(0), -1, -1, -1)], dim=-1)

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
        scores = scores2 + scores1  # bsz x dim x L x L

        scores = self.down_fc(scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return scores


class GlobalPointer(nn.Module):
    def __init__(self, config, inner_dim=200, RoPE=True):
        super().__init__()
        # self.encoder = encoder
        self.ent_type_size = config.w2ner_out
        self.inner_dim = inner_dim
        self.hidden_size = config.bert_config.hidden_size
        if config.use_word_embedding:
            self.hidden_size += config.word_embedding_dim
        if config.use_char_embedding:
            self.hidden_size += config.char_embedding_dim * 2  # bi_lstm
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, word_embedding):

        self.device = word_embedding.device
        batch_size = word_embedding.size()[0]
        seq_len = word_embedding.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(word_embedding)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        return logits


class YoloLayerOutput(nn.Module):
    def __init__(self, config, num_labels):
        super(YoloLayerOutput, self).__init__()
        self.num_labels = num_labels
        self.w2ner_out = config.w2ner_out
        self.yolo_channel = 256
        self.cnn_1 = ComCNN(input_channels=self.w2ner_out, output_channels=self.yolo_channel, kernels=(3, 3), stride=1, padding=1, bn=True, act="leakyrelu")
        self.cnn_2 = nn.Conv2d(in_channels=self.yolo_channel, out_channels=3 * (self.num_labels + 5), kernel_size=(1, 1), stride=1, padding=0)
        self.yolo = YoloLayer([(i, i) for i in config.anchor_size], self.num_labels)

    def forward(self, inputs):
        cnn1 = self.cnn_1(inputs)
        cnn2 = self.cnn_2(cnn1)
        yolo_outputs, yolo_pred = self.yolo(cnn2)

        return yolo_outputs, yolo_pred


class SS2NER(nn.Module):
    def __init__(self, config):
        super(SS2NER, self).__init__()
        self.num_labels = len(config.label_list)
        self.embedding = EmbeddingLayer(config)
        self.GlobalPointer = GlobalPointer(config)
        self.W2NER = W2NER(config)
        self.CNN_Nested = CNN_Nested(config)
        self.yolo_layer1 = YoloLayerOutput(config, self.num_labels)
        self.yolo_layer2 = YoloLayerOutput(config, self.num_labels)
        self.yolo_layer3 = YoloLayerOutput(config, self.num_labels)

    def forward(self, input_ids, word_ids=None, char_ids=None, token_masks_char=None, char_count=None, word_length=None, pieces_index=None, pieces_length=None,
                pieces2word=None, dist_inputs=None, loss_mask=None, labels=None):
        # attention_mask = input_ids.ne(0).float()
        word_embedding, cls_embedding = self.embedding(input_ids, pieces2word, word_ids=word_ids, char_count=char_count, char_ids=char_ids, token_masks_char=token_masks_char)
        global_pointer_output = self.GlobalPointer(word_embedding)
        # yolo_outputs1, yolo_pred1 = self.yolo_layer1(global_pointer_output)

        w2ner_output = self.W2NER(word_embedding, word_length, dist_inputs, loss_mask)
        # yolo_outputs2, yolo_pred2 = self.yolo_layer2(w2ner_output)

        # CNN_Nested_output = self.CNN_Nested(word_embedding, cls_embedding, pieces_index, loss_mask)
        output_conbined = w2ner_output + global_pointer_output  # CNN_Nested_output + global_pointer_output
        yolo_outputs3, yolo_pred3 = self.yolo_layer3(output_conbined)
        # yolo_outputs = torch.cat([yolo_outputs1, yolo_outputs2, yolo_outputs3], dim=1)
        loss = None
        if labels is not None:
            # loss1 = self.compute_loss(yolo_pred1, labels)
            # loss2 = self.compute_loss(yolo_pred2, labels)
            loss3 = self.compute_loss(yolo_pred3, labels)
            loss = loss3
        return {"output": yolo_outputs3, "loss": loss}

    @staticmethod
    def compute_loss(yolo_pred, labels):
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()
        obj_scale, noobj_scale = 1, 100
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=yolo_pred["pred_boxes"],
            pred_cls=yolo_pred["pred_cls"],
            target=labels,
            anchors=yolo_pred["scaled_anchors"],
            ignore_thres=1.0,
        )
        # iou_scores：真实值与最匹配的anchor的IOU得分值 class_mask：分类正确的索引  obj_mask：目标框所在位置的最好anchor置为1 noobj_mask obj_mask那里置0，
        # 还有计算的iou大于阈值的也置0，其他都为1 tx, ty, tw, th, 对应的对于该大小的特征图的xywh目标值也就是我们需要拟合的值 tconf 目标置信度
        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = mse_loss(yolo_pred["x"][obj_mask], tx[obj_mask])  # 只计算有目标的
        loss_y = mse_loss(yolo_pred["y"][obj_mask], ty[obj_mask])
        loss_w = mse_loss(yolo_pred["w"][obj_mask], tw[obj_mask])
        loss_h = mse_loss(yolo_pred["h"][obj_mask], th[obj_mask])
        loss_conf_obj = bce_loss(yolo_pred["pred_conf"][obj_mask], tconf[obj_mask])
        loss_conf_noobj = bce_loss(yolo_pred["pred_conf"][noobj_mask], tconf[noobj_mask])
        loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj  # 有物体越接近1越好 没物体的越接近0越好
        loss_cls = bce_loss(yolo_pred["pred_cls"][obj_mask], tcls[obj_mask])  # 分类损失
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls  # 总损失

        return total_loss


def main():
    print('this message is from main function')


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
