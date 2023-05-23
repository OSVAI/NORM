from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnet_student import resnet8_student, resnet14_student, resnet20_student, resnet32_student, resnet44_student, \
    resnet56_student, resnet110_student, resnet8x4_student, resnet32x4_student

from .resnetv2 import ResNet50

from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .wrn_student import wrn_16_1_student, wrn_16_2_student, wrn_40_1_student, wrn_40_2_student

from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .vgg_student import vgg19_bn_student, vgg16_bn_student, vgg13_bn_student, vgg11_bn_student, vgg8_bn_student

from .mobilenetv2 import mobile_half
from .mobilenetv2_student import mobile_half as mobile_half_student

from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv1_student import ShuffleV1 as ShuffleV1_student

from .ShuffleNetv2 import ShuffleV2
from .ShuffleNetv2_student import ShuffleV2 as ShuffleV2_student



model_channels = {'vgg13': 512,
                  'vgg8': 512,
                  'resnet110': 64,
                  'resnet56': 64,
                  'resnet32': 64,
                  'resnet20': 64,
                  'resnet8': 64,
                  'resnet32x4': 256,
                  'resnet8x4': 256,
                  'ResNet50': 2048,
                  'wrn_40_2': 128,
                  'wrn_40_1': 64,
                  'wrn_16_2': 128,
                  'MobileNetV2': 1280,
                  'ShuffleV1': 960,
                  'ShuffleV2': 1024,
                  }

model_channels_all = {'vgg8': [64, 128, 256,	256, 512],
                      'vgg13': [64, 128, 256,	512, 512],
                      'resnet110': [16,  16,  32,	64],
                      'resnet56': [16,  16,  32,	64],
                      'resnet32': [16,  16,  32,	64],
                      'resnet20': [16,  16,  32,	64],
                      'resnet8': [16,  16,  32,	64],
                      'resnet32x4': [32,  64, 128,  256],
                      'resnet8x4': [32,  64, 128,  256],
                      'ResNet50': [64, 256, 512,  1024, 2048],
                      'wrn_40_2': [16,  32,  64,	128],
                      'wrn_40_1': [16,  16,  32,	64 ],
                      'wrn_16_2': [16,  32,  64,	128],
                      'MobileNetV2': [160, 1280],
                      'ShuffleV1': [24,  240, 680,	960],
                      'ShuffleV2': [24,  116, 232, 1024],
                      }

model_dict_student = {
    'resnet8': resnet8_student,
    'resnet14': resnet14_student,
    'resnet20': resnet20_student,
    'resnet32': resnet32_student,
    'resnet44': resnet44_student,
    'resnet56': resnet56_student,
    # 'resnet110': resnet110_student,
    'resnet8x4': resnet8x4_student,
    'resnet32x4': resnet32x4_student,
    # 'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1_student,
    'wrn_16_2': wrn_16_2_student,
    'wrn_40_1': wrn_40_1_student,
    'wrn_40_2': wrn_40_2_student,
    'vgg8': vgg8_bn_student,
    'vgg11': vgg11_bn_student,
    # 'vgg13': vgg13_bn_student,
    'vgg16': vgg16_bn_student,
    'vgg19': vgg19_bn_student,
    'MobileNetV2': mobile_half_student,
    'ShuffleV1': ShuffleV1_student,
    'ShuffleV2': ShuffleV2_student,
}

model_dict_teacher = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
}
