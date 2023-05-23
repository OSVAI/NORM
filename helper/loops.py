from __future__ import print_function, division

import sys
import time
import torch

import torch
import torch.nn as nn
from .util import AverageMeter, accuracy
import torch.nn.functional as F



def loss_kd(outputs, labels, teacher_outputs, params):
    """
    loss function for Knowledge Distillation (KD)
    """
    alpha = params.alpha
    T = params.temperature

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (T * T)
    KD_loss = (1. - alpha) * loss_CE + alpha * D_KL

    return KD_loss


def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value=alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1 - alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss


def cosine_similarity_loss(x, y):
    loss = 1.0 - F.cosine_similarity(x, y)
    return loss.mean()


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, scheduler=None):
    """One epoch distillation"""
    # # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()
    if opt.dataset == 'cifar100':
        if epoch < 10:
            opt.beta1 = 0.1
        elif epoch < 20:
            opt.beta1 = 0.5
        elif epoch < 180:
            opt.beta1 = 1
        elif epoch < 210:
            opt.beta1 = 0.5
        elif epoch < 240:
            opt.beta1 = 0.1
    else:
        if epoch < 5:
            opt.beta1 = 0.1*1
        elif epoch < 15:
            opt.beta1 = 0.5*1
        elif epoch < 60:
            opt.beta1 = 0.9*1
        elif epoch < 80:
            opt.beta1 = 0.5*1
        elif epoch < 100:
            opt.beta1 = 0.1*1


    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[-2]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_ce = AverageMeter()
    losses_norm = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        # if torch.cuda.is_available():
        input = input.cuda()
        target = target.cuda()
        index = index.cuda()
        if opt.distill in ['crd']:
            contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        logit_s = F.layer_norm(logit_s, torch.Size((opt.num_cls,)), None, None, 1e-7) * opt.ceta
        logit_t = F.layer_norm(logit_t, torch.Size((opt.num_cls,)), None, None, 1e-7) * opt.ceta
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        # other kd beyond KL divergence

        if opt.dataset == 'cifar100':
            # for datset cifar

            if opt.model_s in ['resnet20', 'resnet32', 'resnet8x4', 'wrn_16_2', 'wrn_40_1', 'ShuffleV1', 'ShuffleV2']:
                f_s = feat_s[3]
                f_t = feat_t[3]
            elif opt.model_s in ['MobileNetV2', 'vgg8']:
                f_s = feat_s[4]
                f_t = feat_t[4]
            # print(f_t.shape)
            # print(f_s.shape)
            pool_size = f_t.shape[2]//f_s.shape[2]
            if pool_size > 1:
                f_t = F.max_pool2d(f_t, pool_size, pool_size)

            f_t = f_t.repeat([1, opt.co_sponge, 1, 1])
            loss_norm = F.mse_loss(f_s, f_t.detach()) * opt.co_sponge

            if opt.distill == 'NORM_CRD':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_crd = criterion_kd(f_s, f_t, index, contrast_idx)


        elif opt.dataset == 'imagenet':
            if opt.model_s in ['resnet18S', 'MobileNet', 'resnet50_4S']:
                loss_norm = 0
                # print('teacher feat len:', len(feat_t), ' student_feat len:', len(feat_s))
                # for f_t, f_s in zip(feat_t, feat_s):
                #     print(f_t.shape, f_s.shape)

                for f_t, f_s in zip(feat_t[::-1], feat_s[::-1]): # reversely compute to avoid muti-stage training problem
                    # for f_s in feat_s:\
                    pool_size = f_s.shape[2]//f_t.shape[2]
                    if pool_size > 1:
                        f_s = F.max_pool2d(f_s, pool_size, pool_size)

                    f_t = torch.tile(f_t, [1, opt.co_sponge, 1, 1])

                    loss_norm += F.mse_loss(f_s, f_t.detach()) * opt.co_sponge

            else:
                raise NotImplementedError(opt.model_s)

            if opt.distill == 'NORM_CRD':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_crd = criterion_kd(f_s, f_t, index, contrast_idx)
        else:
            raise NotImplementedError(opt.dataset)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta1 * opt.beta * loss_norm

        if opt.distill == 'NORM_CRD':
            loss += opt.ceta * loss_crd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_ce.update(loss_cls)
        losses_norm.update(loss_norm)
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()
        # print('similarity', NST(feat_t[3].detach(),feat_s[4].detach()))

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, losses_ce.avg, losses_norm.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            if opt.dataset == 'cifar100':
                output = F.layer_norm(output, torch.Size((100,)), None, None, 1e-7) * opt.ceta
            elif opt.dataset == 'imagenet':
                output = F.layer_norm(output, torch.Size((1000,)), None, None, 1e-7) * opt.ceta
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
