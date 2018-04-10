
import os, sys
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def convert_to_one_hot(data, vocab_size, cuda):
    """
        data dims: (batch_size, seq_len)
        returns:(batch_size, seq_len, vocab_size)
    """
    batch_size = data.size(0)
    seq_len = data.size(1)

    samples = Variable(torch.Tensor(batch_size, seq_len, vocab_size))
    one_hot = Variable(torch.zeros((batch_size, vocab_size)).long())

    if(cuda):
        data = data.cuda()
        samples = samples.cuda()
        one_hot = one_hot.cuda()

    for i in range(batch_size):
        x = data[i].view(-1,1)
        one_hot = Variable(torch.zeros((seq_len, vocab_size)).long())
        if cuda:
            one_hot = one_hot.cuda()
        samples[i] = one_hot.scatter_(1, x, 1)

    return samples