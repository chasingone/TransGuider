# -*- coding: utf-8 -*-
# @Author  : Bnaghcneg Zhan
# @File    : TransGuider.py
# @Software: PyCharm
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


logger = logging.getLogger(__name__)

def compute_entropy(x):
    # Flatten The 2D Tensor
    batch_size,num_channels,height,width = x.size()
    flat_feature_map = x.view(batch_size,num_channels,-1)

    channel_entropies = -torch.sum(flat_feature_map * torch.log2(flat_feature_map + 1e-10), dim=2)

    return channel_entropies

class LECA(nn.Module):
    def __init__(self, channel, ratio=16):
        super(LECA, self).__init__()
  
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.norm=nn.BatchNorm2d(channel) 
    def forward(self, x):
  ######################entorypycompute###################
        x1=x.clone()
        chan= int(x1.shape[1])
        box=int(x1.shape[0])

        max_value,_ = torch.max(x1.view(box,chan,-1),dim=2,keepdim=True)
        min_value,_ = torch.min(x1.view(box, chan, -1), dim=2, keepdim=True)
        diff=max_value - min_value
        norm_feature=x1/diff.view(box, chan,1,1)
        x2 = x1 * norm_feature

        entropy=compute_entropy(x2)
        entropy=entropy.view(box,chan,1,1)

        tb=self.norm(entropy)
        entrop_yrate_feature1 =self.sigmoid(tb)

        return  entrop_yrate_feature1

     
class ESRA(nn.Module):
    def __init__(self, channel, droup=0.0):
        super(ESRA, self).__init__()

        self.entory_attention = EntorylAttentionModule(channel)

        self.conv2d = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(normalized_shape = [1,1])
        self.dropout = nn.Dropout(droup)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.num_heads = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):

        entory_value=self.entory_attention(x)*x
        x1=torch.mean(x, dim=1, keepdim=True)
        x2, _=torch.max(x, dim=1, keepdim=True)


        Q=x1.permute(0, 2, 1, 3)
        K=self.conv2d(entory_value).permute(0, 2, 1, 3)
        T=x2.permute(0, 2, 1, 3)
        V=x.permute(0, 2, 1, 3)


        attn1 = Q @ K.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn1 = attn1.softmax(dim=-1)

        attn2 = Q @ T.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn2 = attn2.softmax(dim=-1)

        attn3 = T @ K.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn3 = attn3.softmax(dim=-1)

        attn=attn1+attn2+attn3 

        out = (attn * V).permute(0, 2, 1, 3)
        return out

class SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        xianzhu_feature1 =self.sigmoid(avgout+maxout)*x
        return  xianzhu_feature1

class ACG(nn.Module):
    def __init__(self, channel):
        super(ACG, self).__init__()

        self.Block=ESRA(channel)
        self.norm=nn.BatchNorm2d(channel)
        self.SE=SE(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1=x.clone()
        x1=self.sigmoid(x1)
        x_hardattention = torch.where(x1>0.5,torch.ones_like(x1),torch.zeros_like(x1))
        x_all,entory_value1 = self.Block(x)
        x_SE = self.SE(x)

        x_all=torch.cat([x_hardattention,x_all], dim=1)

        return  x_all,x_SE,entory_value1






