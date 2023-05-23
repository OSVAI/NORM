#!/bin/bash

#train for baseline

for trial in 0 1 2
do

CUDA_VISIBLE_DEVICES=0 python train_cifar_teacher.py --model resnet110 -t $trial|tee save/logs/Teacher_resnet110_t$trial.log &
CUDA_VISIBLE_DEVICES=1 python train_cifar_teacher.py --model resnet56 -t $trial|tee save/logs/Teacher_resnet56_t$trial.log &
CUDA_VISIBLE_DEVICES=2 python train_cifar_teacher.py --model resnet32 -t $trial|tee save/logs/Teacher_resnet32_t$trial.log &
CUDA_VISIBLE_DEVICES=3 python train_cifar_teacher.py --model resnet20 -t $trial|tee save/logs/Teacher_resnet20_t$trial.log &
CUDA_VISIBLE_DEVICES=4 python train_cifar_teacher.py --model resnet8 -t $trial|tee save/logs/Teacher_resnet8_t$trial.log &

CUDA_VISIBLE_DEVICES=5 python train_cifar_teacher.py --model wrn_40_2 -t $trial|tee save/logs/Teacher_wrn_40_2_t$trial.log &
CUDA_VISIBLE_DEVICES=6 python train_cifar_teacher.py --model wrn_40_1 -t $trial|tee save/logs/Teacher_wrn_40_1_t$trial.log &
CUDA_VISIBLE_DEVICES=7 python train_cifar_teacher.py --model wrn_16_2 -t $trial|tee save/logs/Teacher_wrn_16_2_t$trial.log &

CUDA_VISIBLE_DEVICES=3 python train_cifar_teacher.py --model resnet32x4 -t $trial|tee save/logs/Teacher_resnet32x4_t$trial.log &
CUDA_VISIBLE_DEVICES=4 python train_cifar_teacher.py --model resnet8x4 -t $trial|tee save/logs/Teacher_resnet8x4_t$trial.log &

CUDA_VISIBLE_DEVICES=2 python train_cifar_teacher.py --model vgg13 -t $trial|tee save/logs/Teacher_vgg13_t$trial.log &
CUDA_VISIBLE_DEVICES=3 python train_cifar_teacher.py --model vgg8 -t $trial|tee save/logs/Teacher_vgg8_t$trial.log

done


