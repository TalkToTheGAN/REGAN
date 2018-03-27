'''
Auxiliary functions used during training.
'''

import os, sys
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from annex_network import AnnexNetwork
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter

from main import g_sequence_len, BATCH_SIZE, VOCAB_SIZE

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    return Variable(torch.LongTensor(samples[0:batch_size]))

def train_epoch(model, data_iter, criterion, optimizer, cuda=False):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:
    	#tqdm(#data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)

def eval_epoch(model, data_iter, criterion, cuda=False):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:
    	#tqdm(#data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        if cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
    data_iter.reset()
    return math.exp(total_loss / total_words)

# New functions/classes

# return probability distribution (in [0,1]) at the output of G
def g_output_prob(prob):
    softmax = nn.Softmax(dim=1)
    theta_prime = softmax(prob)
    return theta_prime

# performs a Gumbel-Softmax reparameterization of the input
def gumbel_softmax(theta_prime, VOCAB_SIZE, cuda=False):
    u = Variable(torch.log(-torch.log(torch.rand(VOCAB_SIZE))))
    if cuda:
        u = Variable(torch.log(-torch.log(torch.rand(VOCAB_SIZE)))).cuda()
        theta_prime = theta_prime.cuda()
    z = torch.log(theta_prime) - u
    return z

# categorical re-sampling exactly as in Backpropagating through the void - Appendix B
def categorical_re_param(theta_prime, VOCAB_SIZE, b, cuda=False):
    v = Variable(torch.rand(theta_prime.size(0), VOCAB_SIZE))
    z_tilde = Variable(torch.rand(theta_prime.size(0), VOCAB_SIZE))
    if cuda:
        v = v.cuda()
    '''z_tilde = -torch.log(-torch.log(v)/theta_prime - torch.log(v[:,b]))
    z_tilde[:,b] = -torch.log(-torch.log(v[:,b]))'''
    #naive implementation
    for i in range(theta_prime.size(0)):
        v_b = v[i,int(b[i])]
        z_tilde[i,:] = -torch.log((-torch.log(v[i,:])/theta_prime[i,:]) - torch.log(v_b))
        z_tilde[i,int(b[i])]=-torch.log(-torch.log(v[i,int(b[i])]))
    if cuda:
        z_tilde.cuda()

    return z_tilde

# when you have sequences as probability distributions, re-puts them into sequences by doing argmax
def prob_to_seq(x, cuda=False):
    x_refactor = Variable(torch.zeros(x.size(0), x.size(1)))
    if cuda:
        x_refactor = x_refactor.cuda()
    for i in range(x.size(1)):
        x_refactor[:,i] = torch.max(x[:,i,:], 1)[1]
    return x_refactor

#3 e and f : Defining c_phi and getting c_phi(z) and c_phi(z_tilde)
def c_phi_out(GD, c_phi_hat, theta_prime, discriminator, cuda=False):
    # 3.b
    z = gumbel_softmax(theta_prime, VOCAB_SIZE, cuda)
    # 3.c
    value, b = torch.max(torch.transpose(z,0,1),0)
    # 3.d
    z_tilde = categorical_re_param(theta_prime, VOCAB_SIZE, b, cuda)
    if cuda:
        z_tilde = z_tilde.cuda()
    z_gs = gumbel_softmax(z, VOCAB_SIZE, cuda)
    z_tilde_gs = gumbel_softmax(z_tilde, VOCAB_SIZE, cuda)
    # reshaping the inputs of the discriminator
    z_gs = z_gs.view(BATCH_SIZE, g_sequence_len, VOCAB_SIZE)
    z_gs = prob_to_seq(z_gs, cuda)
    z_gs = z_gs.type(torch.LongTensor)
    if cuda:
        z_gs = z_gs.cuda()
    z_tilde_gs = z_tilde_gs.view(BATCH_SIZE, g_sequence_len, VOCAB_SIZE)
    z_tilde_gs = prob_to_seq(z_tilde_gs, cuda)
    z_tilde_gs = z_tilde_gs.type(torch.LongTensor)
    if cuda:
        z_tilde_gs = z_tilde_gs.cuda()
    if GD == 'REBAR':
        return c_phi_hat.forward(z),c_phi_hat.forward(z_tilde)
    if GD == 'RELAX':
        c1=c_phi_hat.forward(z) + discriminator.forward(z_gs)
        c2=c_phi_hat.forward(z_tilde) + discriminator.forward(z_tilde_gs)
        #c2=discriminator.forward(z_tilde_gs)
        if cuda:
            c1=c1.cuda()
            c2=c2.cuda()
    return c1,c2

# get the number of parameters of a neural network
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

'''
Per batch score 
'''
def get_data_goodness_score(all_data):
    # all_data dim: (no_of_sequences, length_of_one_sequence), eeach cell is a string
    total_batch_score = 0
    #for batch_index, batch_input in enumerate(all_data):
    for seq_index, seq_input in enumerate(all_data):
        total_batch_score += get_seq_goodness_score(seq_input)
    return total_batch_score/len(all_data)

def get_seq_goodness_score(seq):
    # seq dim is a string of length len(seq)

    score = 0

    for i in range(len(seq)-2):
        j = i + 3
        sliced_string = seq[i:j]
        
        if sliced_string[0] == 'x' and sliced_string[1]!='x' and sliced_string[2] == 'x':
            score += 1
        elif sliced_string[0] != 'x' and sliced_string[1] =='x' and sliced_string[2] != 'x':
            score+=1
        elif sliced_string[0] == '_' and sliced_string[1] =='_':
            score+=1
        elif sliced_string[1] =='_' and sliced_string[2] == '_':
            score+=1

    return score