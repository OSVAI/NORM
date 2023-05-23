"""
cifar training framework
Only use single GPU training
"""

from __future__ import print_function

import sys
from abc import ABC


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
from models.cifar import model_dict_teacher
from models.cifar import model_dict_student
from models.cifar import model_channels
from models.cifar.util import ConvReg, LinearEmbed
from models.cifar.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from helper.util import adjust_learning_rate
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from distiller_zoo import DistillKL, HintLoss

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tensor board frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epoch', type=int, default=0, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay', type=str, default='step', choices=['step', 'cos'], help='learning decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--num_cls', type=int, default='100', help='num_classes')

    # model
    parser.add_argument('--model_t', type=str, default='resnet110',
                        choices=['wrn_40_2', 'resnet110', 'resnet56', 'resnet32x4', 'vgg13', 'ResNet50'])

    parser.add_argument('--model_s', type=str, default='resnet32',
                        choices=['wrn_16_2', 'wrn_40_1', 'resnet32', 'resnet20', 'resnet8x4', 'vgg8',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])

    parser.add_argument('--path_t', type=str,
                        default="save/models/resnet110_vanilla/resnet110_best.pth",
                        help='teacher model snapshot')

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

    parser.add_argument('--marginal_relu', action='store_true', help='with marginal relu in teacher')

    parser.add_argument('--kd-warm-up', type=float, default=20.0,
                    help='feature konwledge distillation loss weight warm up epochs')
    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'
    opt.dataset = 'cifar100'
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    opt.model_name = f'T_{opt.model_t}_S_{opt.model_s}_cifar100_{opt.distill}_' \
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

    # dataloader
    if opt.distill == 'NORM_CRD':
        train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                           num_workers=opt.num_workers,
                                                                           k=opt.nce_k,
                                                                           mode=opt.mode)
    else:
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)

    # model
    print('==> loading teacher model')
    model_t = model_dict_teacher[opt.model_t](num_classes=opt.num_cls, marginal_relu=opt.marginal_relu)
    model_t.load_state_dict(torch.load(opt.path_t)['model'])
    print('==> done')

    model_s = model_dict_student[opt.model_s](num_classes=opt.num_cls, co_sponge=opt.co_sponge, channel_t=model_channels[opt.model_t])

    module_list = nn.ModuleList([])
    criterion_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])

    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_kd = HintLoss() # MSE loss

    if opt.distill == 'NORM_CRD':
        data = torch.randn(2, 3, 32, 32)
        model_t.eval()
        model_s.eval()
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)

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
    module_list.cuda()
    criterion_list.cuda()

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    time_start = time.time()
    init_epoch = opt.init_epoch
    for epoch in range(init_epoch, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer, len(train_loader))
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, train_loss_ce, trian_loss_norm = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

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
