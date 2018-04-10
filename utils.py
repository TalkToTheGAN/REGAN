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
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from generator import Generator
from discriminator import Discriminator
from annex_network import AnnexNetwork
from data_iter import GenDataIter, DisDataIter
import scipy.stats as stat


from main import GENERATED_NUM, g_sequence_len, BATCH_SIZE, VOCAB_SIZE, SEQ_LEN

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def generate_samples(model, batch_size, generated_num, output_file, cuda=False):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)

    if len(samples) < generated_num:
        delta_samples = model.sample(generated_num - len(samples), g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(delta_samples)
    assert len(samples) == generated_num

    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    gen_samples = Variable(torch.LongTensor(samples[0:batch_size]))
    if cuda:
        gen_samples = gen_samples.cuda()
    return gen_samples

def train_epoch(model, data_iter, criterion, optimizer, PRE_EPOCH_GEN, epoch, cuda=False):
    total_loss = 0.
    total_words = 0.
    i = 0
    # allowing for pre-training on less than an epoch
    dec = PRE_EPOCH_GEN - epoch 
    if (dec > 0) and (dec < 1):
        num_iters = dec * int(GENERATED_NUM / BATCH_SIZE)
    for (data, target) in data_iter:
    	#tqdm(#data_iter, mininterval=2, desc=' - Training', leave=False):
        # print(f"In train_epoch: i (batch number) = {i}")
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
        i += 1
        if (dec > 0) and (dec < 1):
            if i > num_iters:
                break
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


def softmax_with_temp(z, temperature, cuda=False):
    '''
    Peforms softmax with temperature.
    :param z:
    :param temperature:
    :param cuda:
    :return:
    '''
    soft_out = F.softmax(z / temperature, dim=1)
    if cuda:
        soft_out = soft_out.cuda()

    return soft_out

def gumbel_softmax(theta_prime, VOCAB_SIZE, cuda=False):
    '''
    Performs a Gumbel-Softmax reparameterization of the input
    :param theta_prime:
    :param VOCAB_SIZE:
    :param cuda:
    :return:
    '''
    u = Variable(torch.log(-torch.log(torch.rand(VOCAB_SIZE))))
    if cuda:
        u = u.cuda()
        theta_prime = theta_prime.cuda()
    z = torch.log(theta_prime) - u
    return z

# categorical re-sampling exactly as in Backpropagating through the void - Appendix B
def categorical_re_param(theta_prime, VOCAB_SIZE, b, cuda=False):
    v = Variable(torch.rand(theta_prime.size(0), VOCAB_SIZE))
    z_tilde = Variable(torch.rand(theta_prime.size(0), VOCAB_SIZE))
    if cuda:
        v = v.cuda()
    #naive implementation
    for i in range(theta_prime.size(0)):
        v_b = v[i,int(b[i])]
        z_tilde[i,:] = -torch.log((-torch.log(v[i,:])/theta_prime[i,:]) - torch.log(v_b))
        z_tilde[i,int(b[i])]=-torch.log(-torch.log(v[i,int(b[i])]))
    if cuda:
        z_tilde.cuda()
    return z_tilde

# when you have sequences as probability distributions, re-puts them into sequences by doing argmax

def sample_one_hot(theta_prime, batch_size, seq_len, vocab_size, use_cuda):

    # input theta_prime dims = (batch_size * seq_len) x vocab_size
    theta_prime = theta_prime.view(batch_size, seq_len, vocab_size)
    samples = Variable(torch.Tensor(seq_len, batch_size, vocab_size), requires_grad = True)

    if(use_cuda):
        samples = samples.cuda()

    theta_prime = theta_prime.view(seq_len, batch_size, vocab_size)
    for i in range(seq_len):
        x = theta_prime[i].multinomial(1)
        one_hot = Variable(torch.zeros((batch_size, vocab_size)).long(), requires_grad = True)
        if use_cuda:
            one_hot = one_hot.cuda()
        samples[i] = one_hot.scatter_(1, x, 1)

    samples = samples.view(batch_size, seq_len, vocab_size)
    return samples


def prob_to_seq(x, cuda=False):
    batch_size = x.size(0); seq_len = x.size(1); edim = x.size(2)
    print(seq_len)
    x_refactor = Variable(torch.zeros(batch_size, seq_len))
    if cuda:
        x_refactor = x_refactor.cuda()

    for i in range(seq_len):
        x_refactor[:,i] = torch.max(x[:,i,:], 1)[1]
        #test = Categorical(x[:,i,:]).sample().view(x.size(0))
        #x_refactor[:,i] = test

    return x_refactor

#3 e and f : Defining c_phi and getting c_phi(z) and c_phi(z_tilde)
def c_phi_out(GD, c_phi_hat, theta_prime, discriminator, temperature=0.1, eta=None, cuda=False):
    # 3.b
    z = gumbel_softmax(theta_prime, VOCAB_SIZE, cuda)
    # 3.c
    # value, b = torch.max(torch.transpose(z,0,1),0)
    b = sample_one_hot(theta_prime, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda)
    b = b.view(BATCH_SIZE*g_sequence_len, VOCAB_SIZE)
    _, b = torch.max(torch.transpose(b, 0, 1), 0)
    # 3.d
    z_tilde = categorical_re_param(theta_prime, VOCAB_SIZE, b, cuda)
    if cuda:
        z_tilde = z_tilde.cuda()

    # Calculating f(sigmoid_lambda(z)) in Backprop paper.
    type_ = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    f_lambda_z = softmax_with_temp(z, temperature, cuda=cuda)
    f_lambda_z_tilde = softmax_with_temp(z_tilde, temperature, cuda=cuda)

    f_lambda_z = f_lambda_z.view(BATCH_SIZE, g_sequence_len, VOCAB_SIZE)
    f_lambda_z_tilde = f_lambda_z_tilde.view(BATCH_SIZE, g_sequence_len, VOCAB_SIZE)

    # f_lambda_z = prob_to_seq(f_lambda_z, cuda)
    # f_lambda_z_tilde = prob_to_seq(f_lambda_z_tilde, cuda)

    f_lambda_z = f_lambda_z.type(type_)
    f_lambda_z_tilde = f_lambda_z_tilde.type(type_)

    if GD == "REINFORCE":
        if cuda:
            return Variable(torch.zeros((BATCH_SIZE,2)), requires_grad=True).cuda(), Variable(torch.zeros((BATCH_SIZE,2)), requires_grad=True).cuda()
        else:
            return Variable(torch.zeros((BATCH_SIZE,2)), requires_grad=True), Variable(torch.zeros((BATCH_SIZE,2)), requires_grad=True)

    if GD == 'REBAR':
        assert eta is not None
        return eta*discriminator.forward(f_lambda_z), eta*discriminator.forward(f_lambda_z_tilde)

    if (GD == 'RELAX'):
        c1 = torch.add(c_phi_hat.forward(z), discriminator.forward(f_lambda_z))
        c2 = torch.add(c_phi_hat.forward(z_tilde), discriminator.forward(f_lambda_z_tilde))
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
def get_data_goodness_score(all_data, SPACES=False):
    # all_data dim: (no_of_sequences, length_of_one_sequence), eeach cell is a string
    total_batch_score = 0
    for seq_index, seq_input in enumerate(all_data):
        total_batch_score += get_seq_goodness_score(seq_input, SPACES)
    return total_batch_score/len(all_data)

def get_seq_goodness_score(seq, SPACES=False):
    # seq dim is a string of length len(seq)
    if SPACES == True:

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

    else:
        # TODO: There might be issue here with SEQ_LEN = 3, please debug.
        score = 0
        for i in range(len(seq)-2):
            j = i + 3
            sliced_string = seq[i:j]
            
            if sliced_string[0] == 'x' and sliced_string[1]!='x' and sliced_string[2] == 'x':
                score += 1
            elif sliced_string[0] != 'x' and sliced_string[1] =='x' and sliced_string[2] != 'x':
                score+=1

    return score

def get_data_freq(all_data, seq_len=SEQ_LEN):
    # all_data dim: (no_of_sequences, length_of_one_sequence), eeach cell is a string
    if seq_len == 3:
        groundtruth = np.load('freq_array_3.npy')
    elif seq_len == 15:
        groundtruth = np.load('freq_array.npy')

    char_to_ix = {
            'x': 0,
            '+': 1,
            '-': 2,
            '*': 3,
            '/': 4
            # '_': 5,
            #'\n': 6
        }
    batchwise = np.zeros((VOCAB_SIZE,VOCAB_SIZE))
    for seq_index, seq_input in enumerate(all_data):
        for i in range(1,len(seq_input)):
            batchwise[char_to_ix.get(seq_input[i-1]),char_to_ix.get(seq_input[i])]+=1
    sample_size = np.sum(batchwise)
    batchwise = batchwise/sample_size
    
    return stat.entropy(batchwise.reshape(-1),groundtruth.reshape(-1))

def get_char_freq(all_data, SPACES):
    # all_data dim: (no_of_sequences, length_of_one_sequence), eeach cell is a string
    char_to_ix = {
            'x': 0,
            '+': 1,
            '-': 2,
            '*': 3,
            '/': 4
            # '_': 5,
            #'\n': 6
        }
    if SPACES == True:
        batchwise = np.zeros(6)
    else:
        batchwise = np.zeros(5)
    for seq_index, seq_input in enumerate(all_data):
        for i in range(0,len(seq_input)):
            batchwise[char_to_ix.get(seq_input[i])]+=1
  
    return batchwise/(len(all_data)*len(all_data[0]))




