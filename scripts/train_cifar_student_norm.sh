#!/bin/bash
path_teacher_resnet110="./save/models/resnet110_vanilla/resnet110_best.pth"
path_teacher_resnet32x4="./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth"
path_teacher_resnet56="./save/models/resnet56_vanilla/ckpt_epoch_240.pth"
path_teacher_resnet50="./save/models/ResNet50_vanilla/ckpt_epoch_240.pth"
path_teacher_wrn_40_2="./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth"
path_teacher_vgg13="./save/models/vgg13_vanilla/ckpt_epoch_240.pth"

###### same structure ######
#resnet110 -> resnet32
#resnet110 -> resnet20
#resnet56 -> resnet20
#resnet32x4 -> resnet8x4
#wrn-40-2 -> wrn-40-1
#wrn-40-2 -> wrn-16-2
#vgg13 -> vgg8

###### different stucture ######
#vgg13 -> MobileNetV2
#ResNet50 -> MobileNetV2
#ResNet50 -> vgg8
#resnet32x4 -> ShuffleNetV1
#resnet32x4 -> ShuffleNetV2
#wrn-40-2 -> ShuffleNetV1

#take resnet110 -> resnet32 for example
# the training script is like this, the get_gpu.py will automatically
# detection the available GPU and return the GPU number.

for trial in '1' '2' '3'; do
  for r in '0.0625' '0.125' '0.25' '0.5'; do
    for a in '0'; do
      for b in '1';  do
        for s in '8';  do
          tch=resnet110
          stu=resnet32
          pth=${path_teacher_resnet110}
          cuda_num=$(python ./scripts/get_gpu.py)
          echo 'using gpu:' "${cuda_num}"
          echo NORM-${tch}-${stu}-r_${r}-a_${a}-b_${b}-s_${s}-trial_${trial}
          CUDA_VISIBLE_DEVICES=$cuda_num python train_cifar_student.py --model_t ${tch} --model_s ${stu} --path_t ${pth}\
             -r ${r} -a ${a} -b ${b} -s ${s} --trial $trial --lr_decay step \
             2>&1 | tee -a ./save/logs/NORM-${tch}-${stu}-r_${r}-a_${a}-b_${b}-s_${s}-trial_${trial}.log &
          sleep 30s
        done
      done
    done
  done
done

