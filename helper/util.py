from __future__ import print_function

import torch
import numpy as np
from math import cos, pi
import torch.nn as nn

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer, num_iter=0):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    # steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    # if steps > 0:
    #     new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_lr

    warmup_epoch = 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = epoch + epoch * num_iter
    max_iter = opt.epochs * num_iter

    if opt.lr_decay == 'step':
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
    elif opt.lr_decay == 'cos':
        lr = opt.learning_rate * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    print(opt.lr_decay,  lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class LossLabelSmoothing(nn.Module):
    """
    loss function for label smoothing regularization
    """
    def __init__(self):
        super(LossLabelSmoothing, self).__init__()
        self.alpha = 0.1

    def forward(self, outputs, labels):
        N = outputs.size(0)  # batch_size
        C = outputs.size(1)  # number of classes
        smoothed_labels = torch.full(size=(N, C), fill_value= self.alpha / (C - 1)).cuda()
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-self.alpha)

        log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
        loss = -torch.sum(log_prob * smoothed_labels) / N

        return loss

if __name__ == '__main__':

    pass
