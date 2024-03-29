'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )
class MobileNet_(nn.Module):
    def __init__(self, num_classes=1000, co_sponge=8, channel_t=None, **kwargs):
        super(MobileNet_, self).__init__()
        self.fea_dim = 1024
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.model1 = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.model2 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.co_sponge = co_sponge
        if channel_t is not None:
            ch = channel_t
        else:
            ch = 1024
        self.layer31 = conv_1x1_bn(1024, ch*co_sponge)
        self.layer32 = conv_1x1_bn(ch*co_sponge, 1024)

        print("student channels:", ch*co_sponge)
        self.pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, is_feat=False, preact=False, ):
        x = self.model1(x)
        x = self.model2(x)

        # NORM with residual
        res = x
        x = self.layer31(x)  # 16x16
        fea = x
        x = self.layer32(x)  # 16x16
        x += res

        x = self.pool(x)
        x = x.view(-1, 1024)
        f1 = x
        x = self.fc(x)
        if is_feat:
            return [fea,], x
        else:
            return x

def MobileNet(num_classes=1000, co_sponge=8, channel_t=None,**kwargs):
    model = MobileNet_(num_classes=num_classes, co_sponge=co_sponge, channel_t=channel_t, **kwargs)
    return model

if __name__ == '__main__':
    net_G = MobileNet()
    sub_params = sum(p.numel() for p in net_G.parameters())
    print(sub_params)
    # net(torch.randn(2, 3, 224, 224))


