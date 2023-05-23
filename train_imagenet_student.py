"""
Imagenet general training framework
This use multi-GPU training

"""

from __future__ import print_function
from collections import OrderedDict
import sys
from abc import ABC


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

import os
import argparse
import socket
import time
import datetime

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from models.imagenet import model_dict_imagenet_teacher, model_dict_imagenet_student
from models.imagenet import model_channels_imagenet

from dataset.imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate



def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=500, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=0, help='init training for two-stage methods')
    parser.add_argument('--gids', type=str, default='0,1,2,3,4,5,6,7', help='save frequency')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay', type=str, default='step', choices=['step', 'cos'], help='learning decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='30, 60, 90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--num_cls', type=int, default='1000', help='num_classes')
    # model
    parser.add_argument('--model_t', type=str, default=None,
                        choices=[None, 'resnet50', 'resnet34',  'resnet18', 'MobileNet', 'resnet50_4'])
    parser.add_argument('--model_s', type=str, default='resnet18S', choices=['resnet18S', 'MobileNet', 'resnet50_4S'])

    parser.add_argument('--use_layer3', action='store_true', help='use the features to learn!')
    #parser.add_argument('--use_layer4', action='store_true', help='use the features to learn!')
    parser.add_argument('--marginal_relu', action='store_true', help='with marginal relu in teacher')


    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot, if None, load from modelzoo.')
    parser.add_argument('--resume_path', type=str, default=None, help='path_to resume weights!')

    # distillation
    parser.add_argument('--distill', type=str, default='NORM', choices=['NORM', 'NORM_CRD'])
    parser.add_argument('--trial', type=str, default='1', help='trial tag')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='the weight of the standard CE loss based on the ground truth labels of training data.')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='the hyper-parameter /beta in our paper of NORM, weighting the vanilla KD loss defined as the KL divergence between the teacher and student logits.')
    parser.add_argument('-b', '--beta', type=float, default=None, help=' the hyper-parameter /alpha in our paper of NORM, weighting the NORM loss')
    parser.add_argument('-b1', '--beta1', type=float, default=1, help='weight balance for other losses')
    parser.add_argument('-c', '--ceta', type=float, default=2.5, help='weight balance for other losses')
    parser.add_argument('-s', '--co-sponge', type=int, default=4, help='the hyper-parameter in our paper of NORM')


    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    parser.add_argument('--kd-warm-up', type=float, default=20.0,
                    help='feature konwledge distillation loss weight warm up epochs')
    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'
    opt.dataset = 'imagenet'
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = f'T_{opt.model_t}_S_{opt.model_s}_{opt.dataset}_{opt.distill}_' \
        f'r_{opt.gamma}_a_{opt.alpha}_b_{opt.beta}_c_{opt.ceta}_s_{opt.co_sponge}_{opt.trial}'

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def main():
    best_acc = 0
    opt = parse_option()
    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    train_loader, val_loader, n_data = get_imagenet_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)
    opt.num_cls = 1000
    print(opt)

    # model and use multi-GPUs
    gids = opt.gids.split(',')
    gpu_ids = list([])
    for it in gids:
        gpu_ids.append(int(it))

    print('==> loading teacher model: ', opt.model_t)
    # by default load the model zoo pre-trained weights
    model_t = model_dict_imagenet_teacher[opt.model_t](num_classes=opt.num_cls, marginal_relu=opt.marginal_relu, model_path=opt.model_path)
    model_t = torch.nn.DataParallel(model_t,  device_ids=gpu_ids).cuda().eval()

    model_s = model_dict_imagenet_student[opt.model_s](num_classes=opt.num_cls, co_sponge=opt.co_sponge, channel_t=model_channels_imagenet[opt.model_t], use_layer3=opt.use_layer3)
    model_s = torch.nn.DataParallel(model_s,  device_ids=gpu_ids).cuda().train()

    if opt.resume_path is not None:
        weights = torch.load(opt.resume_path)
        model_s.load_state_dict(weights['model'])
        opt.init_epochs = weights['epoch'] + 1
        print(datetime.datetime.now())
        print(f"start from save epoch: {weights['epoch']}, best acc: {weights['best_acc']}")

    print("teacher name:", opt.model_t, "teacher channels:", model_channels_imagenet[opt.model_t])
    print("student name:", opt.model_s)


    #append student network into module_list
    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])
    criterion_list = nn.ModuleList([])

    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    # criterion_cls = LossLabelSmoothing() # this is the smooth label Loss
    criterion_div = DistillKL(opt.kd_T)
    criterion_kd = HintLoss()  # MSE loss


    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_s)
    module_list.append(model_t)
    # module_list.cuda()
    criterion_list.cuda()

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    time_start = time.time()
    # routine
    init_epochs = opt.init_epochs
    for epoch in range(init_epochs, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer, len(train_loader))
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, train_loss_ce, trian_loss_norm = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, (time2 - time1)/60.0))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)
            print('best accuracy:', best_acc)
        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    time_end = time.time()
    print("total_cost:", time_end - time_start)
    print('best accuracy:', best_acc)
    print('time:{}'.format(time.asctime(time.localtime(time.time()) ))+'  ')
    with open('save/results.txt', 'a') as f:
        tm = time.localtime(time.time())
        f.write("{:0>4}{:0>2}{:0>2}{:0>2}{:0>2}".format(tm[0], tm[1], tm[2], tm[3], tm[4]))
        # f.write(f"{tm[0]:0>4}{tm[1]:0>2}{tm[2]:0>2}{tm[3]:0>2}{tm[4]:0>2}")
        f.write(opt.model_name+'  ')
        f.write('best_accuracy:{} '.format(best_acc))
        f.write("total_cost:{}\n".format(time_end - time_start))

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
