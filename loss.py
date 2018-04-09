# -*- coding: utf-8 -*-
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
    
    # def old_forward(self, GD, prob, target, reward, c_phi_hat, discriminator, BATCH_SIZE, g_sequence_len, cuda=False):
    #     """
    #     Forward function with implementation based on the original one.
    #     """
    #     N = target.size(0)
    #     C = prob.size(1)
    #     one_hot = torch.zeros((N, C))
    #     if cuda:
    #         one_hot = one_hot.cuda()
    #     one_hot.scatter_(1, target.data.view((-1,1)), 1)
    #     one_hot = one_hot.type(torch.ByteTensor)
    #     one_hot = Variable(one_hot)
    #     if cuda:
    #         one_hot = one_hot.cuda()
    #     loss = torch.masked_select(prob, one_hot)
    #     loss = loss.view(BATCH_SIZE, g_sequence_len)
    #     loss = torch.mean(loss, 1)
    #     c_phi_z, c_phi_z_tilde = utils.c_phi_out(GD, c_phi_hat, prob, discriminator, cuda)
    #     c_phi_z_tilde = c_phi_z_tilde[:,1]
    #     c_phi_z = c_phi_z[:,1]
    #     loss = loss * (reward - c_phi_z_tilde) + c_phi_z - c_phi_z_tilde
    #     loss =  - torch.sum(loss)
    #     return loss

    # def forward(self, GD, prob, samples, reward, c_phi_hat, discriminator, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda=False):
    #     """
    #     Computes the Generator's loss in RELAX optimization setting.
    #
    #     """
    #     prob_temp = prob.view(BATCH_SIZE, g_sequence_len, VOCAB_SIZE)
    #     new_prob = Variable(torch.zeros(BATCH_SIZE, g_sequence_len))
    #     if cuda:
    #         new_prob = new_prob.cuda()
    #     for i in range(BATCH_SIZE):
    #         for j in range(g_sequence_len):
    #             new_prob[i,j] = prob_temp[i,j,int(samples[i,j])]
    #     loss = torch.sum(new_prob, 1)
    #     c_phi_z, c_phi_z_tilde = utils.c_phi_out(GD, c_phi_hat, prob, discriminator, cuda)
    #     c_phi_z_tilde = c_phi_z_tilde[:,1]
    #     c_phi_z = c_phi_z[:,1]
    #     loss = loss * (reward - c_phi_z_tilde) + c_phi_z - c_phi_z_tilde
    #     loss =  - torch.sum(loss)
    #     return loss

    def forward_reward(self, i, samples, prob, rewards, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda=False):
        """
        Computes the Generator's loss in RELAX optimization setting. 

        """
        conditional_proba = Variable(torch.zeros(BATCH_SIZE, VOCAB_SIZE))
        if cuda:
            conditional_proba = conditional_proba.cuda()
        for j in range(BATCH_SIZE):
            #conditional_proba[j, int(samples[j, i])] = prob[j, i, int(samples[j, i])]
            conditional_proba[j, int(samples[j, i])] = 1
            conditional_proba[j, :] = - (rewards[j] * conditional_proba[j, :]) / BATCH_SIZE 
        return conditional_proba

    def forward_reward_grads(self, samples, prob, rewards, g, BATCH_SIZE, g_sequence_len, VOCAB_SIZE, cuda=False):
        """
        Computes the Generator's loss in RELAX optimization setting. 

        """
        conditional_proba = Variable(torch.zeros(BATCH_SIZE, g_sequence_len, VOCAB_SIZE))
        batch_grads = []
        if cuda:
            conditional_proba = conditional_proba.cuda()
        for j in range(BATCH_SIZE):
            for i in range(g_sequence_len):
                #conditional_proba[j, i, int(samples[j, i])] = prob[j, i, int(samples[j, i])]
                conditional_proba[j, i, int(samples[j, i])] = 1
            conditional_proba[j, :, :] = - (rewards[j] * conditional_proba[j, :, :])
        #print(conditional_proba[0, :, :])
        for j in range(BATCH_SIZE):
            j_grads = []
            g.zero_grad()
            prob[j, :, :].backward(conditional_proba[j, :, :], retain_graph=True)
            for p in g.parameters():
                j_grads.append(p.grad)
            batch_grads.append(j_grads)
        #print(batch_grads[0][6])
        return batch_grads

class VarianceLoss(nn.Module):
    """Loss for the control variate annex network"""
    def __init__(self):
        super(VarianceLoss, self).__init__()

    def forward(self, grad, cuda = False):
        # bs = len(grad)
        # square_sum = 0
        # normal_sum = 0
        # print("GRAD SHAPE----------------", len(grad))
        # print('--------------------------', len(grad[0]))
        # print('--------------------------')
        # #print(grad[0][1].size())
        # #print(grad[0][1][0])
        # #print(grad[0][1][0].size())
        # for j in range(bs):
        #     for i in range(len(grad[j])):
        #         # print(grad[j][i].shape)
        #         # print(grad[j][i])
        #         square_sum += (torch.sum(grad[j][i]**2).data[0])/bs
        #         normal_sum += (torch.sum(grad[j][i]).data[0])/bs
        #         # print(square_sum)
        #         # print(normal_sum)

        # print('------------------------------------')
        # print(square_sum)
        # print(normal_sum)
        # var_2nd_term = (normal_sum)**2
        # print("Variance:", square_sum - var_2nd_term)
        # total_loss = np.array([square_sum])
        # total_loss = Variable(torch.Tensor(total_loss), requires_grad=True)
        # total_variance = total_loss.data[0] - var_2nd_term
        # # print(total_loss.data[0])
        # # print(var_2nd_term)
        # if cuda:
        #     total_loss = total_loss.cuda()
        #     total_variance = total_variance.cuda()
        # return total_loss, total_variance
        bs = len(grad)
        ref = 0
        for j in range(bs):
            for i in range(len(grad[j])):
                ref += torch.sum(grad[j][i]**2).item()
        total_loss = np.array([ref/bs])
        total_loss = Variable(torch.Tensor(total_loss), requires_grad=True)
        print(total_loss)
        if cuda:
            total_loss = total_loss.cuda()
        return total_loss

    def forward_variance(self, grad, cuda=False):

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



