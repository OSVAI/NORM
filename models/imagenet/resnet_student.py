import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_1x1_bn_relu(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, low_dim=None, div=1, co_sponge=8, channel_t=None,use_layer3=False, **kwargs):
        super(ResNet, self).__init__()
        zero_init_residual = False
        groups = 1
        width_per_group = 64
        replace_stride_with_dilation = None
        norm_layer = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.list = [2048]
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64//div, layers[0])
        self.layer2 = self._make_layer(block, 128//div, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256//div, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512//div, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.co_sponge = co_sponge
        if channel_t is not None:
            ch = channel_t
        else:
            ch = 512*block.expansion//div
        self.use_layer4 = True
        self.layer41 = conv_1x1_bn(512*block.expansion//div, ch*co_sponge)
        print("student channels:", ch*co_sponge)
        self.layer42 = conv_1x1_bn(ch*co_sponge, 512*block.expansion//div)

        self.use_layer3 = use_layer3
        if self.use_layer3:
            ch = 256 # this setting is temporary for alignment of feature layer3 of resnet 50
            self.layer31 = conv_1x1_bn(256*block.expansion//div, ch*co_sponge)
            self.layer32 = conv_1x1_bn(ch*co_sponge, 256*block.expansion//div)

        self.layer21 = conv_1x1_bn(128*block.expansion//div, ch*co_sponge)
        self.layer22 = conv_1x1_bn(ch*co_sponge, 128*block.expansion//div)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if low_dim>0:
            self.low_dim = low_dim
            self.fea_dim = low_dim
            self.fc = nn.Linear(512//div * block.expansion, low_dim)
        else:
            self.fea_dim = 512//div * block.expansion
            self.fc = nn.Linear(512//div * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self.dropout = nn.Dropout(p=0.3)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, feat_s=None, preact=False):
        if not feat_s is None:
            x = self.avgpool(feat_s)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        else:
            fea_return = []
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)

            x = self.layer3(x)
            if self.use_layer3:
                if self.co_sponge == 1:
                    fea = x
                else:
                    res = x
                    x = self.layer31(x)
                    fea = x
                    x = self.layer32(x)
                    x += res
                fea_return.append(fea)
            x = self.layer4(x)
            if self.use_layer4:
                if self.co_sponge == 1:
                    fea = x
                else:
                    res = x
                    x = self.layer41(x)
                    # x = self.dropout(x)
                    fea = x
                    x = self.layer42(x)
                    x += res
                fea_return.append(fea)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            if is_feat:
                return fea_return, x
            else:
                return x

def resnet18S(num_classes=1000, low_dim=0, co_sponge=8, channel_t=None,**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, low_dim=low_dim, co_sponge=co_sponge, channel_t=channel_t,**kwargs)
    return model

def resnet50_4S(num_classes=1000, low_dim=0, co_sponge=8, channel_t=None,**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, low_dim=low_dim, div=4, co_sponge=co_sponge, channel_t=channel_t,**kwargs)
    return model

def resnet50S(num_classes=1000, low_dim=0, co_sponge=8, channel_t=None,**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, low_dim=low_dim, co_sponge=co_sponge, channel_t=channel_t,**kwargs)
    return model

def resnet34S(num_classes=1000, low_dim=0, co_sponge=8, channel_t=None,**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, low_dim=low_dim, co_sponge=co_sponge, channel_t=channel_t,**kwargs)
    return model



if __name__ == '__main__':
    net_G = resnet34S()
    sub_params = sum(p.numel() for p in net_G.parameters())
    print(sub_params)