# the training script is like this. You can modify them as you need

path_teacher_resnet34="./save/models/resnet34_imagenet_vanilla/resnet34-333f7ec4.pth"
path_teacher_resnet50="./save/models/resnet50_imagenet_vanilla/resnet50-19c8e357.pth"

#resnet34 -> resnet18
for trial in 'cos_decay_0'; do
  for r in '1'; do
    for a in '0'; do
      for b in '1'; do
      s=8
      tch=resnet34
      stu=resnet18S
      path_tch=${path_teacher_resnet34}
#      resume_path='save/student_model/S:resnet18S_T:resnet34_imagenet_hint_r:0.5_a:0.5_b:0.5_c:2.5_s:8_1_kdt_4/resnet18S_best.pth'
      python train_student_imagenet.py --model_t ${tch} --model_s ${stu} --path_t ${path_tch}  \
      -r ${r} -a ${a} -b ${b} -s ${s} --trial $trial --gids=0,1,2,3,4,5,6,7  --lr_decay cos 2>&1 | tee ./save/logs/NORM-ImageNet-${tch}-${stu}-r_${r}-a_${a}-b_${b}-s_${s}-trial_${trial}.log
      echo "save/logs/NORM-ImageNet-${tch}-${stu}-${r}_${a}_${b}_${s}_trial_${trial}.log"
      sleep 30s
      done
    done
  done
done

#resnet50 -> MobileNet
for trial in e; do
  for r in '0.125'; do
    for a in '0'; do
      for b in '1'; do
      s=8
      tch=resnet50
      stu=MobileNet
      path_tch=${path_teacher_resnet50}
      echo "save/logs/MSD-ImageNet-${tch}-${stu}-${r}_${a}_${b}_${s}_trial_${trial}_decay_cos.log"
#      resume_path='save/student_model/S:MobileNet_T:resnet50_imagenet_hint_r:0.5_a:0.0_b:4.0_c:2.5_s:8_d_kdt_4/MobileNet_best.pth'
#      echo "resume_path=save/student_model/S:MobileNet_T:resnet50_imagenet_hint_r:0.5_a:0.0_b:4.0_c:2.5_s:8_d_kdt_4/MobileNet_best.pth"
      python train_student_noLS.py --dataset imagenet --path_t ${path_tch} --distill hint --model_s ${stu}  -r ${r} -a ${a} -b ${b} -s ${s} --trial $trial --gids=0,1,2,3,4,5,6,7 --lr_decay cos 2>&1 | tee -a ./save/logs/MSD-ImageNet-${tch}-${stu}-${r}_${a}_${b}_${s}_trial_${trial}_decay_cos.log

      sleep 30s
      done
    done
  done
done
