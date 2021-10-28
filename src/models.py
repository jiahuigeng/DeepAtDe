import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from src.variation_bottleneck import VariationalBottleneck
import os.path as osp

class LeNet(nn.Module):
    def __init__(self, vb, channel, img_shape,  hidden=None,num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        if not hidden:
            hidden = int(img_shape[0] * img_shape[1] * 3 /4)
        self.vb = vb
        self.hidden = hidden
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.vb_layer = VariationalBottleneck((hidden,))
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        fc_in = out.view(out.size(0), -1)
        if self.vb:
            vb_fc_in = self.vb_layer(fc_in)
            fc_out = self.fc(vb_fc_in)
        else:
            fc_out = self.fc(fc_in)
        return fc_out, fc_in


class MLP(nn.Module):
    def __init__(self, vb, channel, img_shape, hidden, num_classes):
        super(MLP, self).__init__()
        if not hidden:
            hidden = 1024
        self.hidden = hidden
        self.vb = vb
        self.vb_layer = VariationalBottleneck((hidden,))
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel*img_shape[0]*img_shape[1], hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        fc_in = out.view(out.size(0), -1)
        if self.vb:
            vb_fc_in = self.vb_layer(fc_in)
            fc_out = self.fc(vb_fc_in)
        else:
            fc_out = self.fc(fc_in)
        return fc_out, fc_in

    #     width = 2**(args.data_scale+5)
    #     input_dim = width * width
    #     hidden =  int(width * width / 256)
    #     act = nn.Sigmoid
    #
    #     self.body = nn.Sequential(
    #         nn.Linear(input_dim, hidden),
    #         act(),
    #         # nn.Linear(hidden, hidden),
    #         # act(),
    #     )
    #
    #     self.fc = nn.Sequential(nn.Linear(hidden*3, num_classes))
    #
    # def forward(self, x):
    #     x = x.view(x.shape[0], x.shape[1], -1)
    #     out = self.body(x)
    #     fc_in = out.view(out.size(0), -1)
    #     fc_out = self.fc(fc_in)
    #     return fc_out, fc_in


def weights_init(m):
    torch.manual_seed(8)
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class ResNet18(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        base = resnet.resnet18(pretrained=self.pretrained)
        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool)
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, self.num_classes, bias=True)
        for name, module in self.named_modules():
            if hasattr(module, 'relu'):
                module.relu = nn.Sigmoid()

    def forward(self, x):
        h = self.in_block(x)
        h = self.encoder1(h)
        h = self.encoder2(h)
        h = self.encoder3(h)
        h = self.encoder4(h)
        fc_in = torch.flatten(self.avgpool(h), 1)
        fc_out = self.fc(fc_in)
        return fc_out, fc_in