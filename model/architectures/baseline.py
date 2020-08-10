import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from model.backbones.isqrt import MPNCOV
from model.aggregations.spoc import SPoC
from model.aggregations.crow import Crow
from model.aggregations.rmac import RMAC
from model.aggregations.gem import GeM
from model.backbones.resnet50 import ResNet50_Pre_Trained


class My_Res50(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(My_Res50, self).__init__()

        self.resnet = models.resnet50(pretrained)  # resnst 50
        # self.resnet = models.resnet101(pretrained)  # resnst 101

        # Feature extractor
        self.layer1 = nn.Sequential(*list(self.resnet.children())[:-5])  # output 56*56 feature map
        self.layer2 = nn.Sequential(*list(self.resnet.children())[-5])  # output 28*28 feature map
        self.layer3 = nn.Sequential(*list(self.resnet.children())[-4])  # output 14*14 feature map
        self.layer4 = nn.Sequential(*list(self.resnet.children())[-3])  # output 7*7 feature map

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, epoch):
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

        gap_fea = self.gap(layer4_fea)
        logit = self.fc(gap_fea.view(gap_fea.shape[0], -1))
        # gap_fea = self.gap(layer4_fea).view(layer4_fea.shape[0], -1)
        # gmp_fea = self.gmp(layer4_fea).view(layer4_fea.shape[0], -1)
        # final_fea = torch.cat([gap_fea, gmp_fea], dim=1)
        # logit = self.fc(final_fea)

        return logit


class My_Res50_isqrt(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(My_Res50_isqrt, self).__init__()

        self.resnet = models.resnet50(pretrained)  # resnst 50

        # Feature extractor
        self.layer1 = nn.Sequential(*list(self.resnet.children())[:-5])  # output 56*56 feature map
        self.layer2 = nn.Sequential(*list(self.resnet.children())[-5])  # output 28*28 feature map
        self.layer3 = nn.Sequential(*list(self.resnet.children())[-4])  # output 14*14 feature map
        self.layer4 = nn.Sequential(*list(self.resnet.children())[-3])  # output 7*7 feature map

        self.isqrt = MPNCOV(iterNum=5, is_sqrt=True, is_vec=True, input_dim=2048, dimension_reduction=256)
        self.fc = nn.Linear(32896, num_classes, bias=False)

    def forward(self, x, epoch):
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

        isqrt_fea = self.isqrt(layer4_fea)
        isqrt_fea = isqrt_fea.view(isqrt_fea.shape[0], -1)
        logit = self.fc(isqrt_fea)

        return logit


class My_FPN(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(My_FPN, self).__init__()

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

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1, 1))

        self.fc_1 = nn.Linear(256, num_classes)
        self.fc_2 = nn.Linear(256, num_classes)
        self.fc_3 = nn.Linear(256, num_classes)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear')
        return z + y

    def forward(self, x, epoch):
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
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        # p1 = self.smooth4(p1)

        gap_p4 = self.gap(p4)
        gap_p3 = self.gap(p3)
        gap_p2 = self.gap(p2)
        p4_logit = self.fc_1(gap_p4.view(gap_p4.shape[0], -1))
        p3_logit = self.fc_2(gap_p3.view(gap_p3.shape[0], -1))
        p2_logit = self.fc_3(gap_p2.view(gap_p2.shape[0], -1))

        # return (p4_logit + p3_logit + p2_logit) / 3
        return p4_logit, p3_logit, p2_logit


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


class Old_My_Counting_Net_Insight_Select(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(Old_My_Counting_Net_Insight_Select, self).__init__()

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

        # meta selection
        self.selection_net = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.se_fc = nn.Linear(256, 3)

        # Classifiers
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1, 1))

        self.fc_1 = nn.Linear(256, num_classes)
        self.fc_2 = nn.Linear(256, num_classes)
        self.fc_3 = nn.Linear(256, num_classes)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear')
        return z + y

    def forward(self, x, epoch):
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
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        p1 = self.smooth4(p1)

        # select head
        se_fea = torch.cat([p4, self.avgpool2d(p3), self.avgpool2d(self.avgpool2d(p2))], dim=1)
        se_fea = self.selection_net(se_fea)
        select_weight = torch.softmax(self.se_fc(se_fea.view(se_fea.shape[0], -1)), dim=1)

        gap_p4 = self.gap(p4)
        gap_p3 = self.gap(p3)
        gap_p2 = self.gap(p2)

        p4_logit = self.fc_1(gap_p4.view(gap_p4.shape[0], -1))
        p3_logit = self.fc_2(gap_p3.view(gap_p3.shape[0], -1))
        p2_logit = self.fc_3(gap_p2.view(gap_p2.shape[0], -1))

        if epoch >= 4:
            p4_logit = p4_logit * select_weight[:, 0].view(select_weight.shape[0], 1)
            p3_logit = p3_logit * select_weight[:, 1].view(select_weight.shape[0], 1)
            p2_logit = p2_logit * select_weight[:, 2].view(select_weight.shape[0], 1)

            final_logit = p4_logit + p3_logit + p2_logit
        else:
            final_logit = (p4_logit + p3_logit + p2_logit) / 3

        return final_logit, select_weight


class Old_My_Counting_Net_Insight_Gated(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(Old_My_Counting_Net_Insight_Gated, self).__init__()

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

        # meta selection
        self.gated_net = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.ga_fc = nn.Linear(256, 3)

        # Classifiers
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1, 1))

        self.fc_1 = nn.Linear(256, num_classes)
        self.fc_2 = nn.Linear(256, num_classes)
        self.fc_3 = nn.Linear(256, num_classes)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear')
        return z + y

    def forward(self, x, epoch):
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
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        p1 = self.smooth4(p1)

        # select head
        ga_fea = torch.cat([p4, self.avgpool2d(p3), self.avgpool2d(self.avgpool2d(p2))], dim=1)
        ga_fea = self.gated_net(ga_fea)
        gated_weight = self.ga_fc(ga_fea.view(ga_fea.shape[0], -1))
        gated_weight = torch.softmax(gated_weight, dim=1)

        gap_p4 = self.gap(p4)
        gap_p3 = self.gap(p3)
        gap_p2 = self.gap(p2)

        p4_logit = self.fc_1(gap_p4.view(gap_p4.shape[0], -1))
        p3_logit = self.fc_2(gap_p3.view(gap_p3.shape[0], -1))
        p2_logit = self.fc_3(gap_p2.view(gap_p2.shape[0], -1))

        p4_logit = p4_logit * gated_weight[:, 0].view(gated_weight.shape[0], 1)
        p3_logit = p3_logit * gated_weight[:, 1].view(gated_weight.shape[0], 1)
        p2_logit = p2_logit * gated_weight[:, 2].view(gated_weight.shape[0], 1)

        final_logit = p4_logit + p3_logit + p2_logit

        return final_logit
