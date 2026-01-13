#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : layers.py
@Author  : HuYing
@Time    : 2022/12/28 18:40
@Description: 
"""
import torch.nn as nn
import torch
import numpy as np


class LinearActivation(nn.Module):
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, input):
        return input


ACT2FN = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.CELU(),
    "leakyrelu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "linear": LinearActivation(),
    "selu": nn.SELU(),
    "elu": nn.ELU()
}


class MLP(nn.Module):
    """
    全连接层，dropout-->全连接-->激活
    """

    def __init__(self, n_in, n_out, act="leakyrelu", dropout=0.0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        if act is not None:
            self.activation = ACT2FN[act]
        else:
            self.activation = None
        if dropout != 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MultiHeadBiaffine(nn.Module):
    def __init__(self, dim, out=None, n_head=4):
        super(MultiHeadBiaffine, self).__init__()
        assert dim % n_head == 0
        in_head_dim = dim // n_head
        out = dim if out is None else out
        assert out % n_head == 0
        out_head_dim = out // n_head
        self.n_head = n_head
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_head, out_head_dim, in_head_dim, in_head_dim)))
        self.out_dim = out

    def forward(self, h, v):
        """

        :param h: bsz x max_len x dim
        :param v: bsz x max_len x dim
        :return: bsz x max_len x max_len x out_dim
        """
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.n_head, -1)
        v = v.reshape(bsz, max_len, self.n_head, -1)
        w = torch.einsum('blhx,hdxy,bkhy->bhdlk', h, self.W, v)
        w = w.reshape(bsz, self.out_dim, max_len, max_len)
        return w


class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False, groups=groups)

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class W2nerLayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(W2nerLayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, act="gelu", dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.act = ACT2FN[act]
        self.base = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(input_size, channels, kernel_size=1), nn.GELU(), )
        self.convs = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = self.act(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class W2NERConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, act="leakyrelu", dropout=0.1):
        super(W2NERConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )
        self.act = ACT2FN[act]
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = self.act(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class MaskCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3):
        super(MaskCNN, self).__init__()

        layers = []
        for i in range(depth):
            layers.extend([
                MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU()])
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x  # 用作residual
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x


class PMaskCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3):
        super(PMaskCNN, self).__init__()

        layers = []
        for i in range(depth):
            layers.extend([
                MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU()
                ]
            )
        #layers.append(MaskConv2d(input_channels, output_channels, kernel_size=3, padding=3//2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x  # 用作residual
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x


def seq_len_to_mask(seq_len, max_len=None):
    r"""

    将一个表示 ``sequence length`` 的一维数组转换为二维的 ``mask`` ，不包含的位置为 **0**。
    :param seq_len: 大小为 ``(B,)`` 的长度序列；
    :param int max_len: 将长度补齐或截断到 ``max_len``。默认情况（为 ``None``）使用的是 ``seq_len`` 中最长的长度；
        但在 :class:`torch.nn.DataParallel` 等分布式的场景下可能不同卡的 ``seq_len`` 会有区别，所以需要传入
        ``max_len`` 使得 ``mask`` 的补齐或截断到该长度。
    :return: 大小为 ``(B, max_len)`` 的 ``mask``， 元素类型为 ``bool`` 或 ``uint8``
    """
    max_len = int(max_len) if max_len is not None else int(seq_len.max())

    if isinstance(seq_len, np.ndarray):
        assert seq_len.ndim == 1, f"seq_len can only have one dimension, got {seq_len.ndim}."
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
        return mask

    try:  # 尝试是否是 torch
        if isinstance(seq_len, torch.Tensor):
            assert seq_len.ndim == 1, f"seq_len can only have one dimension, got {seq_len.ndim == 1}."
            batch_size = seq_len.shape[0]
            broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
            mask = broad_cast_seq_len < seq_len.unsqueeze(1)
            return mask
    except NameError as e:
        pass

    raise TypeError("seq_len_to_mask function only supports numpy.ndarray, torch.Tensor, paddle.Tensor, "
                    f"jittor.Var and oneflow.Tensor, but got {type(seq_len)}")


class ComCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernels=(3, 3), stride=1, padding=1, bn=True, act="leakyrelu"):
        super(ComCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernels, stride=stride, padding=padding,
                              bias=not bn)
        if bn:
            self.bn = nn.BatchNorm2d(output_channels, momentum=0.9, eps=1e-5)
        else:
            self.bn = None
        if act is not None:
            self.activation = ACT2FN[act]
        else:
            self.activation = None

        # 参数初始化。不这么初始化，容易梯度爆炸nan
        self.conv.weight.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(output_channels, input_channels, kernels[0], kernels[1])))
        # self.bn.weight.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(output_channels,)))
        # self.bn.bias.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(output_channels,)))
        # self.bn.running_mean.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(output_channels,)))
        # self.bn.running_var.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(output_channels,)))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes=6, img_dim=80):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 1
        self.obj_scale = 1
        self.noobj_scale = 100
        self.grid_size = 0  # grid size
        self.img_dim = img_dim

    def compute_grid_offsets(self, grid_size, cuda=None):
        device = torch.device("cuda" if cuda else "cpu")
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).float().to(device)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).float().to(device)
        self.scaled_anchors = torch.tensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]).float().to(device)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x):
        num_samples = x.size(0)
        grid_size = x.size(2)
        self.img_dim = x.size(2)
        prediction = (x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous())
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)  # 相对位置得到对应的绝对位置比如之前的位置是0.5,0.5变为 11.5，11.5这样的

        # Add offset and scale with anchors #特征图中的实际位置
        pred_boxes = torch.ones(prediction[..., :4].shape).float().to(x.device)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat((pred_boxes.view(num_samples, -1, 4) * self.stride, pred_conf.view(num_samples, -1, 1),
                            pred_cls.view(num_samples, -1, self.num_classes),), -1)
        return output, dict(
            x=x,
            y=y,
            w=w,
            h=h,
            pred_conf=pred_conf,
            pred_cls=pred_cls,
            pred_boxes=pred_boxes,
            scaled_anchors=self.scaled_anchors
        )


def main():
    print('this message is from main function')


if __name__ == '__main__':
    main()
    print('now __name__ is %s' % __name__)
