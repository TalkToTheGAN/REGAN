'''
Loss functions.
'''

import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 

import utils


class NLLLoss(nn.Module):
    """Self-Defined NLLLoss Function

    Args:
        weight: Tensor (num_class, )
    """
    def __init__(self, weight):
        super(NLLLoss, self).__init__()
        self.weight = weight

    def forward(self, prob, target):
        """
        Args:
            prob: (N, C) 
            target : (N, )
        """
        N = target.size(0)
        C = prob.size(1)
        weight = Variable(self.weight).view((1, -1))
        weight = weight.expand(N, C)  # (N, C)
        if prob.is_cuda:
            weight = weight.cuda()
        prob = weight * prob

        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        
        return -torch.sum(loss)


class GANLoss(nn.Module):

    """Reward-Refined NLLLoss Function for adversial training of Generator"""

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward_reinforce(self, prob, target, reward, cuda=False):
        """
        Forward function used in the SeqGAN implementation. 
        Args:
            prob: (N, C), torch Variable 
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)

        return loss
    
    def forward_reward(self, i, samples, prob, rewards, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda=False):
        """
        Returns what is used to get the gradient contribution of the i-th term of the batch.

        """
        conditional_proba = Variable(torch.zeros(BATCH_SIZE, VOCAB_SIZE))
        if cuda:
            conditional_proba = conditional_proba.cuda()
        for j in range(BATCH_SIZE):
            conditional_proba[j, int(samples[j, i])] = 1
            conditional_proba[j, :] = - (rewards[j]/BATCH_SIZE * conditional_proba[j, :]) 

        return conditional_proba

    def forward_reward_grads(self, samples, prob, rewards, g, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda=False):
        """
        Returns a list of gradient contribution of every term in the batch

        """
        conditional_proba = Variable(torch.zeros(BATCH_SIZE, g_sequence_len, VOCAB_SIZE))
        batch_grads = []
        if cuda:
            conditional_proba = conditional_proba.cuda()
        for j in range(BATCH_SIZE):
            for i in range(g_sequence_len):
                conditional_proba[j, i, int(samples[j, i])] = 1
            conditional_proba[j, :, :] = - (rewards[j] * conditional_proba[j, :, :])
        for j in range(BATCH_SIZE):
            j_grads = []
            # since we want to isolate each contribution, we have to zero the generator's gradients here. 
            g.zero_grad()
            prob[j, :, :].backward(conditional_proba[j, :, :], retain_graph=True)
            for p in g.parameters():
                j_grads.append(p.grad.clone())
            batch_grads.append(j_grads)

        return batch_grads

class VarianceLoss(nn.Module):

    """Loss for the control variate annex network"""

    def __init__(self):
        super(VarianceLoss, self).__init__()

    def forward(self, grad, cuda = False):
        """
        Used to get the gradient of the variance. 

        """
        bs = len(grad)
        ref = 0
        for j in range(bs):
            for i in range(len(grad[j])):
                ref += torch.sum(grad[j][i]**2).item()
        total_loss = np.array([ref/bs])
        total_loss = Variable(torch.Tensor(total_loss), requires_grad=True)
        if cuda:
            total_loss = total_loss.cuda()

        return total_loss

    def forward_variance(self, grad, cuda=False):
        """
        Used to get the variance of one single parameter. 
        In this case, we take look at the last layer, then take the variance of the first parameter of this last layer in main.py

        """
        bs = len(grad)
        n_layers = len(grad[0])
        square_term = torch.zeros((grad[0][n_layers-1].size()))
        normal_term = torch.zeros((grad[0][n_layers-1].size()))
        if cuda:
            square_term = square_term.cuda()
            normal_term = normal_term.cuda()
        for j in range(bs):
            square_term = torch.add(square_term, grad[j][n_layers-1]**2)
            normal_term = torch.add(normal_term, grad[j][n_layers-1])
        square_term /= bs
        normal_term /= bs
        normal_term = normal_term ** 2

        return square_term - normal_term



