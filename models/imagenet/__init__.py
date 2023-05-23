from .resnet import resnet34T, resnet50T, resnet18T
from .MobileNet import MobileNet
from .resnet_student import resnet50_4S, resnet34S, resnet18S

model_dict_imagenet_teacher = {
    'resnet18': resnet18T,
    'resnet50': resnet50T,
    'resnet34': resnet34T,}

model_dict_imagenet_student = {
    'resnet34S': resnet34S,
    'resnet18S': resnet18S,
    'MobileNet': MobileNet,
    'resnet50_4S': resnet50_4S,
}

model_channels_imagenet = {
                  'resnet18': 512,
                  'resnet34': 512,
                  'resnet50': 2048,
                  }

