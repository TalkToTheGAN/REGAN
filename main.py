# -*- coding:utf-8 -*-

import os
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

from utils import *
from loss import *


isDebug = True

# ================== Parameter Definition =================

# Basic Training Parameters
THC_CACHING_ALLOCATOR=0
SEED = 88
random.seed(SEED)
np.random.seed(SEED)
BATCH_SIZE = 16
GENERATED_NUM = 10000
# related to data
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 5000
# pre-training
PRE_EPOCH_GEN = 1 if isDebug else 120
PRE_EPOCH_DIS = 1 if isDebug else 5
PRE_ITER_DIS = 1 if isDebug else 3
# adversarial training
GD = 'RELAX' # REBAR or RELAX
UPDATE_RATE = 0.8
TOTAL_BATCH = 100
G_STEPS = 1 if isDebug else 1
D_STEPS = 4 if isDebug else 4
D_EPOCHS = 2 if isDebug else 2
# Generator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 20
# Discriminator Parameters
d_emb_dim = 64
#d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
#d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_filter_sizes = [1, 2, 3, 4]
d_num_filters = [100, 200, 200, 200]
d_dropout = 0.75
d_num_class = 2
# Annex network parameters
#c_filter_sizes = [1, 3, 5, 7, 9, 15]
#c_num_filters = [12, 25, 25, 12, 12, 20]
c_filter_sizes = [1, 3]
c_num_filters = [12, 25]

def main(opt):

    cuda = opt.cuda; visualize = opt.visualize
    print(f"cuda = {cuda}, visualize = {opt.visualize}")

    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, cuda)
    n_gen = Variable(torch.Tensor([get_n_params(generator)]))
    if cuda:
        n_gen = n_gen.cuda()
    print(n_gen)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    c_phi_hat = AnnexNetwork(d_num_class, VOCAB_SIZE, d_emb_dim, c_filter_sizes, c_num_filters, d_dropout, BATCH_SIZE, g_sequence_len)
    target_lstm = TargetLSTM(VOCAB_SIZE, g_emb_dim, g_hidden_dim, cuda)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        c_phi_hat = c_phi_hat.cuda()
        target_lstm = target_lstm.cuda()

    # Generate toy data using target lstm
    print('Generating data ...')
    generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE)
    
    # Load data from file
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)

    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(size_average=False)
    gen_optimizer = optim.Adam(generator.parameters())
    if cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_GEN):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer, cuda)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        loss = eval_epoch(target_lstm, eval_iter, gen_criterion, cuda)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Discriminator ...')
    for epoch in range(PRE_EPOCH_DIS):
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for _ in range(PRE_ITER_DIS):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer, cuda)
            print('Epoch [%d], loss: %f' % (epoch, loss))

    # Adversarial Training 
    rollout = Rollout(generator, UPDATE_RATE)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(size_average=False)
    if cuda:
        gen_criterion = gen_criterion.cuda()
    
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())
    if cuda:
        dis_criterion = dis_criterion.cuda()
    
    c_phi_hat_loss = VarianceLoss()
    if cuda:
        c_phi_hat_loss = c_phi_hat_loss.cuda()
    c_phi_hat_optm = optim.Adam(c_phi_hat.parameters())
    
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(G_STEPS):
            samples = generator.sample(BATCH_SIZE, g_sequence_len)
            # samples has size (BS, sequence_len)
            # construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            if cuda:
                rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
            # rewards has size (BS)
            prob = generator.forward(inputs)
            # prob has size (BS*sequence_len, VOCAB_SIZE)
            # 3.a
            theta_prime = g_output_prob(prob)
            # theta_prime has size (BS*sequence_len, VOCAB_SIZE)
            # 3.e and f
            c_phi_z_ori, c_phi_z_tilde_ori = c_phi_out(GD, c_phi_hat, theta_prime, discriminator, cuda)
            c_phi_z = torch.sum(c_phi_z_ori[:,1])
            c_phi_z_tilde = -torch.sum(c_phi_z_tilde_ori[:,1])
            # 3.i
            grads = []
            first_term_grads = []
            # 3.h optimization step
            # first, empty the gradient buffers
            gen_gan_optm.zero_grad()
            # first, re arrange prob
            new_prob = prob.view((BATCH_SIZE, g_sequence_len, VOCAB_SIZE))
            # 3.g new gradient loss for relax 
            batch_i_grads_1 = gen_gan_loss.forward_reward_grads(samples, new_prob, rewards, generator, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda)
            batch_i_grads_2 = gen_gan_loss.forward_reward_grads(samples, new_prob, c_phi_z_tilde_ori[:,1], generator, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda)
            # batch_i_grads should be of length BATCH SIZE of arrays of all the gradients
            # # 3.i
            batch_grads = batch_i_grads_1
            for i in range(len(batch_i_grads_1)):
                for j in range(len(batch_i_grads_1[i])):
                    batch_grads[i][j] += batch_i_grads_2[i][j]
            # batch_grads should be of length BATCH SIZE
            grads.append(batch_grads)
            # For loop that computes gradients to update G
            generator.zero_grad()
            for i in range(g_sequence_len):
                # 3.g new gradient loss for relax 
                cond_prob = gen_gan_loss.forward_reward(i, samples, new_prob, rewards, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda)
                c_term = gen_gan_loss.forward_reward(i, samples, new_prob, c_phi_z_tilde_ori[:,1], BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda)
                cond_prob -= c_term
                new_prob[:, i, :].backward(cond_prob, retain_graph=True)
            # 3.h 
            c_phi_z.backward(retain_graph=True)
            c_phi_z_tilde.backward(retain_graph=True)
            gen_gan_optm.step()
            # 3.i
            # c_phi_z term
            partial_grads = []
            for j in range(BATCH_SIZE):
                generator.zero_grad()
                c_phi_z_ori[j,1].backward(retain_graph=True)
                j_grads = []
                for p in generator.parameters():
                    j_grads.append(p.grad)
                partial_grads.append(j_grads)
            grads.append(partial_grads)
            # c_phi_z_tilde term
            partial_grads = []
            for j in range(BATCH_SIZE):
                generator.zero_grad()
                c_phi_z_tilde_ori[j,1].backward(retain_graph=True)
                j_grads = []
                for p in generator.parameters():
                    j_grads.append(p.grad)
                partial_grads.append(j_grads)
            grads.append(partial_grads)
            # grads should be of length 3
            # grads[0] should be of length BATCH SIZE
            # 3.j
            all_grads = grads[0]
            for i in range(len(grads[0])):
                for j in range(len(grads[0][i])):
                    all_grads[i][j] += grads[1][i][j] + grads[2][i][j]
            # all_grads should be of length BATCH_SIZE
            c_phi_hat_optm.zero_grad()
            var_loss = c_phi_hat_loss.forward(all_grads, cuda)/n_gen
            var_loss.backward()
            c_phi_hat_optm.step()
            print('Batch {} estimate of the variance of the gradient: {}'.format(total_batch, var_loss.data[0]))

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
            loss = eval_epoch(target_lstm, eval_iter, gen_criterion, cuda)
            print('Batch %d True Generator Loss: %f' % (total_batch, loss))
        
        for a in range(D_STEPS):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for b in range(D_EPOCHS):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer, cuda)
                print('Batch {} Discriminator Loss at step {} and epoch {}: {}'.format(total_batch, a, b, loss))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument('--visualize', action='store_true', help='Enables Visdom')
    parser.add_argument('--cuda', action='store', default=None, type=int)
    opt = parser.parse_args()
    if opt.cuda is not None and opt.cuda >= 0:
        torch.cuda.set_device(opt.cuda)
        opt.cuda = True if torch.cuda.is_available() else False

    try:
        from eval.helper import *
        from eval.BLEU_score import *
        import torchnet as tnt
        from torchnet.engine import Engine
        from torchnet.logger import VisdomPlotLogger, VisdomLogger
        canVisualize = True
    except ImportError as ie:
        eprint("Could not import vizualization imports. ")
        canVisualize = False

    opt.visualize = True if (opt.visualize and canVisualize) else False
    main(opt)
