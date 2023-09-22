import math
import torch
import torch.nn as nn
from torch.nn import init
from ReIDModules.AIM_CCReID.models.ResNet import resnet50

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)


def pcb_block(num_ftrs, num_stripes, local_conv_out_channels, feature_dim, avg=False):
    if avg:
        pooling_list = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(num_stripes)])
    else:
        pooling_list = nn.ModuleList([nn.AdaptiveMaxPool2d(1) for _ in range(num_stripes)])
    conv_list = nn.ModuleList([nn.Conv2d(num_ftrs, local_conv_out_channels, 1, bias=False) for _ in range(num_stripes)])
    batchnorm_list = nn.ModuleList([nn.BatchNorm2d(local_conv_out_channels) for _ in range(num_stripes)])
    relu_list = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(num_stripes)])
    for m in conv_list:
        weight_init(m)
    for m in batchnorm_list:
        weight_init(m)
    return pooling_list, conv_list, batchnorm_list, relu_list


def spp_vertical(feats, pool_list, conv_list, bn_list, relu_list, num_strides, feat_list=[]):
    for i in range(num_strides):
        pcb_feat = pool_list[i](feats[:, :, i * int(feats.size(2) / num_strides): (i + 1) * int(feats.size(2) / num_strides), :])
        pcb_feat = conv_list[i](pcb_feat)
        pcb_feat = bn_list[i](pcb_feat)
        pcb_feat = relu_list[i](pcb_feat)
        pcb_feat = pcb_feat.view(pcb_feat.size(0), -1)
        feat_list.append(pcb_feat)
    return feat_list


def global_pcb(feats, pool, conv, bn, relu, feat_list=[]):
    global_feat = pool(feats)
    global_feat = conv(global_feat)
    global_feat = bn(global_feat)
    global_feat = relu(global_feat)
    global_feat = global_feat.view(feats.size(0), -1)
    feat_list.append(global_feat)
    return feat_list


class PM(nn.Module):
    def __init__(self, feature_dim, blocks = 15, num_stripes=6, local_conv_out_channels=256, avg=False, **kwargs):
        super(PM, self).__init__()
        self.num_stripes = num_stripes

        model_ft = resnet50(pretrained=True, last_conv_stride=1)
        self.num_ftrs = list(model_ft.layer4)[-1].conv1.in_channels
        self.features = model_ft

        self.global_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_conv = nn.Conv2d(self.num_ftrs, local_conv_out_channels, 1, bias=False)
        self.global_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.global_relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(256 * blocks, feature_dim, bias=False)
        self.bn = nn.BatchNorm1d(feature_dim)

        weight_init(self.global_conv)
        weight_init(self.global_bn)
        weight_init(self.linear)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.pcb2_pool_list, self.pcb2_conv_list, self.pcb2_batchnorm_list, self.pcb2_relu_list = pcb_block(self.num_ftrs, 2, local_conv_out_channels, feature_dim, avg)
        self.pcb4_pool_list, self.pcb4_conv_list, self.pcb4_batchnorm_list, self.pcb4_relu_list = pcb_block(self.num_ftrs, 4, local_conv_out_channels, feature_dim, avg)
        self.pcb8_pool_list, self.pcb8_conv_list, self.pcb8_batchnorm_list, self.pcb8_relu_list = pcb_block(self.num_ftrs, 8, local_conv_out_channels, feature_dim, avg)

    def forward(self, x):
        feats = self.features(x)

        feat_list = global_pcb(feats, self.global_pooling, self.global_conv, self.global_bn, self.global_relu, [])
        feat_list = spp_vertical(feats, self.pcb2_pool_list, self.pcb2_conv_list, self.pcb2_batchnorm_list, self.pcb2_relu_list, 2, feat_list)
        feat_list = spp_vertical(feats, self.pcb4_pool_list, self.pcb4_conv_list, self.pcb4_batchnorm_list, self.pcb4_relu_list, 4, feat_list)
        feat_list = spp_vertical(feats, self.pcb8_pool_list, self.pcb8_conv_list, self.pcb8_batchnorm_list, self.pcb8_relu_list, 8, feat_list)

        f = torch.cat(feat_list, dim=1)
        f = self.linear(f)
        f = self.bn(f)

        return feats, f