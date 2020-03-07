#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for Toward Adversarial Robustness via Semi-supervised Robust Training
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable



def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()



def SRTD_loss(model,
                x_l,
                y_l,
                x_nl,
                optimizer,
                IsSemi,
                criterion=nn.CrossEntropyLoss(),
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                lambada=1.0,
                distance='l_inf'):
    
    model.eval()
    if IsSemi == True:
        x_mix = torch.cat((x_l,x_nl),0) #Get the D'_L + D_{NL}
    else:
        x_mix = x_l
    batch_size = len(x_mix)


    #Get the prediction of labeled data and nonlabeled data
    prediction_mix= model(x_mix)
    targets_mix = torch.argmax(prediction_mix, dim=1)
    
    
    # Generate 'adversarial' examples with respect to robustness
    x_mix_adv = x_mix.detach() + 0.001 * torch.randn(x_mix.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_mix_adv.requires_grad_()
            with torch.enable_grad():
                loss_rob = criterion(model(x_mix_adv),targets_mix)
            grad = torch.autograd.grad(loss_rob, [x_mix_adv])[0]
            x_mix_adv = x_mix_adv.detach() + step_size * torch.sign(grad.detach())
            x_mix_adv = torch.min(torch.max(x_mix_adv, x_mix - epsilon), x_mix + epsilon)
            x_mix_adv = torch.clamp(x_mix_adv, 0.0, 1.0)
    elif distance == 'l_2':
        for _ in range(perturb_steps):
            x_mix_adv.requires_grad_()
            with torch.enable_grad():
                loss_rob = criterion(model(x_mix_adv),targets_mix)
            grad = torch.autograd.grad(loss_rob, [x_mix_adv])[0]
            x_mix_adv = x_mix_adv.detach()
            for idx_batch in range(batch_size):
                grad_idx = grad[idx_batch]
                grad_idx_norm = l2_norm(grad_idx)
                grad_idx /= (grad_idx_norm + 1e-8)
                x_mix_adv[idx_batch] = x_mix_adv[idx_batch].detach() + step_size * grad_idx
                eta_x_mix_adv = x_mix_adv[idx_batch] - x_mix[idx_batch]
                mix_eta = l2_norm(eta_x_mix_adv)
                if mix_eta > epsilon:
                    eta_x_mix_adv = eta_x_mix_adv * epsilon / l2_norm(eta_x_mix_adv)
                x_mix_adv[idx_batch] = x_mix[idx_batch] + eta_x_mix_adv
            x_mix_adv = torch.clamp(x_mix_adv, 0.0, 1.0)
    else:
        x_mix_adv = torch.clamp(x_mix_adv, 0.0, 1.0)


    #Define the SRTD loss
    model.train()
    x_mix_adv = Variable(torch.clamp(x_mix_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate different components
    loss_stand = criterion(model(x_l), y_l)
    loss_robust = criterion(model(x_mix_adv), targets_mix)
    loss = loss_stand + lambada * loss_robust
    return loss





