# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class AnnexNetwork(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout, batch_size, g_sequence_len):
        super(AnnexNetwork, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, vocab_size)) for (n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.g_sequence_len = g_sequence_len
        self.init_parameters()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size*g_sequence_len, vocab_size)
        """
        emb = x.view(self.batch_size, 1, self.g_sequence_len, self.vocab_size) # batch_size * 1 * seq_len * vocab_size
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))

        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.normal_(0, 0.02)


