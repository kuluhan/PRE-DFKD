import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10, CIFAR100, FashionMNIST, ImageFolder

def eval_model(net, data_test, data_test_loader):
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            '''
            if epoch == 1:
                disp = images[:49]
                save_image(disp.detach(), './figure/FASHION.png', nrow=7, normalize=True)
            '''
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += torch.nn.CrossEntropyLoss()(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    return accr

def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 30:
        lr = learing_rate
    elif epoch < 80:
        lr = 0.1*learing_rate
    elif epoch < 1000:
        lr = 0.01*learing_rate
    else:
        lr = lr = 0.001*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lr(optimizer, epoch, learing_rate):
    if epoch < 50:
        lr = learing_rate
    elif epoch < 100:
        lr = 0.1*learing_rate
    else:
        lr = lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lr_MNIST(optimizer, epoch, learing_rate):
    if epoch < 5:
        lr = learing_rate
    elif epoch < 10:
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def JS_divergence(outputs, outputs_student):
    T = 3.0
    # Jensen Shanon divergence:
    # another way to force KL between negative probabilities
    P = nn.functional.softmax(outputs_student / T, dim=1)
    Q = nn.functional.softmax(outputs / T, dim=1)
    M = 0.5 * (P + Q)

    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    eps = 0.0
    loss_verifier_cig = 0.5 * nn.KLDivLoss()(torch.log(P + eps), M) + 0.5 * nn.KLDivLoss()(torch.log(Q + eps), M)
    # JS criteria - 0 means full correlation, 1 - means completely different
    loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
    return loss_verifier_cig

def kdloss(y, teacher_scores,T=1.0):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) *(T**2)  / y.shape[0]
    return l_kl