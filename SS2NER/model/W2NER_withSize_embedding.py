#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : baseline.py
@Author  : HuYing
@Time    : 2022/12/27 17:06
@Description:
"""
from model.layers import *
from transformers import AutoModel
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
    target = target[target[:, 1] > 0]  # 去除没有实体的句子
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
        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hid_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hid_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

        size_embed_dim = 25
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

        self.W = torch.nn.Parameter(torch.empty(cls_num, hsz))
        torch.nn.init.xavier_normal_(self.W.data)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)  # bsz x L x L x dim

        # z = self.dropout(self.mlp_rel(z))
        # o2 = self.linear(z)  # bsz x L x L x dim

        head_state = self.head_mlp(x)
        tail_state = self.tail_mlp(x)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)

        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:x.size(1), :x.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(x.size(0), -1, -1, -1)], dim=-1)
        o3 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L

        out = o1.permute(0, 3, 1, 2) + o3
        #out = out.permute(0, 3, 1, 2) + o3
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

        word_reps = torch.cat(embeds, dim=-1)
        return word_reps


class W2NER(nn.Module):
    def __init__(self, config):
        super(W2NER, self).__init__()
        self.lstm_input_size = config.bert_config.hidden_size
        if config.use_word_embedding:
            self.lstm_input_size += config.word_embedding_dim
        if config.use_char_embedding:
            self.lstm_input_size += config.char_embedding_dim * 2  # bi_lstm
        self.biaffine_size = config.biaffine_size
        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size
        self.dilation = config.dilation
        self.ffnn_hid_size = config.ffnn_hid_size
        self.w2ner_dropout = config.w2ner_dropout
        self.w2ner_out = config.w2ner_out
        self.w2ner_conv_dropout = config.w2ner_conv_dropout
        self.conv_input_size = self.lstm_hid_size + config.dist_emb_size + config.type_emb_size
        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)

        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hid_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.convLayer = W2NERConvolutionLayer(self.conv_input_size, self.conv_hid_size, self.dilation, act="leakyrelu", dropout=self.w2ner_conv_dropout)
        self.predictor = CoPredictor(self.w2ner_out, self.lstm_hid_size, self.biaffine_size, self.conv_hid_size * len(self.dilation), self.ffnn_hid_size,
                                     self.w2ner_dropout)  # B, C, H, W
        self.cln = W2nerLayerNorm(self.lstm_hid_size, self.lstm_hid_size, conditional=True)

    def forward(self, word_reps, word_length, dist_inputs, loss_mask):
        packed_embs = pack_padded_sequence(word_reps, word_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.lstm(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=word_length.max())

        cln = self.cln(word_reps.unsqueeze(2), word_reps)

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(loss_mask.clone().long())
        reg_inputs = tril_mask + loss_mask.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, loss_mask.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, loss_mask.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)

        return outputs


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
        self.W2NER = W2NER(config)
        self.yolo_layer = YoloLayerOutput(config, self.num_labels)

    def forward(self, input_ids, word_ids=None, char_ids=None, token_masks_char=None, char_count=None, word_length=None, pieces_index=None, pieces_length=None,
                pieces2word=None, dist_inputs=None, loss_mask=None, labels=None):
        word_embedding = self.embedding(input_ids, pieces2word, word_ids=word_ids, char_count=char_count, char_ids=char_ids, token_masks_char=token_masks_char)
        w2ner_output = self.W2NER(word_embedding, word_length, dist_inputs, loss_mask)
        yolo_outputs, yolo_pred = self.yolo_layer(w2ner_output)

        loss = None
        if labels is not None:
            loss = self.compute_loss(yolo_pred, labels)
        return {"output": yolo_outputs, "loss": loss}

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
        total_loss = 10 * loss_x + 10 * loss_y + 10 * loss_w + 10 * loss_h + loss_conf + 50 * loss_cls  # 总损失

        return total_loss


def main():
    print('this message is from main function')


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
