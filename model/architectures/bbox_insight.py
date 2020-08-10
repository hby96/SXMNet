import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from model.backbones.resnet50 import ResNet50_Pre_Trained


def cut_fuse_block(input_dim, part_out_dim, reduce_factor):
    block = nn.Sequential(
        nn.Conv2d(input_dim, part_out_dim, 3, padding=1),
        nn.BatchNorm2d(part_out_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(part_out_dim, int(part_out_dim / reduce_factor), 1),
        nn.BatchNorm2d(int(part_out_dim / reduce_factor)),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(part_out_dim / reduce_factor), part_out_dim, 1),
        nn.BatchNorm2d(part_out_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(part_out_dim, part_out_dim, 3, padding=1),
        nn.BatchNorm2d(part_out_dim),
        nn.ReLU(inplace=True),
    )
    return block


def _make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class My_Counting_Net_Insight(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(My_Counting_Net_Insight, self).__init__()

        self.resnet = models.resnet50(pretrained)  # resnst 50

        # Feature extractor
        self.layer1 = nn.Sequential(*list(self.resnet.children())[:-5])  # output 56*56 feature map
        self.layer2 = nn.Sequential(*list(self.resnet.children())[-5])  # output 28*28 feature map
        self.layer3 = nn.Sequential(*list(self.resnet.children())[-4])  # output 14*14 feature map
        self.layer4 = nn.Sequential(*list(self.resnet.children())[-3])  # output 7*7 feature map

        # Top layer
        self.top_layer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)

        # Counting layers
        self.down_sample = nn.Conv2d(256, 128, kernel_size=3, stride=2, dilation=2, padding=2)
        self.backend_feat_1 = [128, 128, 128, 64]
        self.backend_1 = _make_layers(self.backend_feat_1, in_channels=131, dilation=True)
        self.output_layer_1 = nn.Conv2d(64, 1, kernel_size=1)

        self.backend_feat_2 = [32, 32, 32, 16]
        self.backend_2 = _make_layers(self.backend_feat_2, in_channels=4, dilation=True)
        self.output_layer_2 = nn.Conv2d(16, 1, kernel_size=1)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear')
        return z + y

    def forward(self, x):
        """
            Returns:
              local_feat_list: each member with shape [N, c]
              logits_list: each member with shape [N, num_classes]
            """

        # Bottom-up
        layer1_fea = self.layer1(x)
        layer2_fea = self.layer2(layer1_fea)
        layer3_fea = self.layer3(layer2_fea)  # shape [N, C, H, W]
        layer4_fea = self.layer4(layer3_fea)

        # Top-down
        p4 = self.top_layer(layer4_fea)
        p3 = self.upsample_add(p4, self.latlayer1(layer3_fea))
        p2 = self.upsample_add(p3, self.latlayer2(layer2_fea))
        p1 = self.upsample_add(p2, self.latlayer3(layer1_fea))

        # Smooth
        p1 = self.smooth4(p1)

        x_112 = self.avgpool2d(x)
        x_56 = self.avgpool2d(x_112)
        x_28 = self.avgpool2d(x_56)
        x_14 = self.avgpool2d(x_28)
        x_7 = self.avgpool2d(x_14)

        # Counting head
        ht_map_1 = self.down_sample(p1)
        ht_map_1 = self.backend_1(torch.cat([x_28, ht_map_1], dim=1))
        ht_map_1 = self.output_layer_1(ht_map_1)
        ht_map_1 = torch.sigmoid(ht_map_1)

        ht_map_2 = F.upsample(ht_map_1, size=(p1.shape[2], p1.shape[3]), mode='bilinear')
        ht_map_2 = self.backend_2(torch.cat([x_56, ht_map_2], dim=1))
        ht_map_2 = self.output_layer_2(ht_map_2)
        ht_map_2 = torch.sigmoid(ht_map_2)

        return ht_map_1, ht_map_2
