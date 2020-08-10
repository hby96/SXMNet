import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from model.backbones.resnet50 import ResNet50_Pre_Trained


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

        # self.down_sample_28 = nn.Conv2d(256, 128, kernel_size=3, stride=2, dilation=2, padding=2)
        # self.backend_feat_28 = [128, 128, 128, 64]
        # self.backend_28 = _make_layers(self.backend_feat_28, in_channels=131, dilation=True)
        # self.output_layer_28 = nn.Conv2d(64, 1, kernel_size=1)
        #
        # self.down_sample_14 = nn.Conv2d(256, 128, kernel_size=3, stride=2, dilation=2, padding=2)
        # self.backend_feat_14 = [128, 128, 128, 64]
        # self.backend_14 = _make_layers(self.backend_feat_14, in_channels=132, dilation=True)
        # self.output_layer_14 = nn.Conv2d(64, 1, kernel_size=1)
        #
        # self.down_sample_7 = nn.Conv2d(256, 128, kernel_size=3, stride=2, dilation=2, padding=2)
        # self.backend_feat_7 = [128, 128, 128, 64]
        # self.backend_7 = _make_layers(self.backend_feat_7, in_channels=132, dilation=True)
        # self.output_layer_7 = nn.Conv2d(64, 1, kernel_size=1)

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

        ht_map_2 = F.upsample(ht_map_1, size=(56, 56), mode='bilinear')
        ht_map_2 = self.backend_2(torch.cat([x_56, ht_map_2], dim=1))
        ht_map_2 = self.output_layer_2(ht_map_2)
        ht_map_2 = torch.sigmoid(ht_map_2)

        # Attention class head
        ht_map_28 = self.avgpool2d(ht_map_2)
        ht_map_14 = self.avgpool2d(ht_map_28)
        ht_map_7 = self.avgpool2d(ht_map_14)

        # Attention class head
        gap_p4 = self.gap(p4 * ht_map_7)
        gap_p3 = self.gap(p3 * ht_map_14)
        gap_p2 = self.gap(p2 * ht_map_28)
        p4_logit = self.fc_1(gap_p4.view(gap_p4.shape[0], -1))
        p3_logit = self.fc_2(gap_p3.view(gap_p3.shape[0], -1))
        p2_logit = self.fc_3(gap_p2.view(gap_p2.shape[0], -1))

        return (p4_logit + p3_logit + p2_logit) / 3


class My_Counting_Net_Insight_Select(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(My_Counting_Net_Insight_Select, self).__init__()

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

        self.down_sample_28 = nn.Conv2d(256, 128, kernel_size=3, stride=2, dilation=2, padding=2)
        self.backend_feat_28 = [128, 128, 128, 64]
        self.backend_28 = _make_layers(self.backend_feat_28, in_channels=131, dilation=True)
        self.output_layer_28 = nn.Conv2d(64, 1, kernel_size=1)

        self.down_sample_14 = nn.Conv2d(256, 128, kernel_size=3, stride=2, dilation=2, padding=2)
        self.backend_feat_14 = [128, 128, 128, 64]
        self.backend_14 = _make_layers(self.backend_feat_14, in_channels=132, dilation=True)
        self.output_layer_14 = nn.Conv2d(64, 1, kernel_size=1)

        self.down_sample_7 = nn.Conv2d(256, 128, kernel_size=3, stride=2, dilation=2, padding=2)
        self.backend_feat_7 = [128, 128, 128, 64]
        self.backend_7 = _make_layers(self.backend_feat_7, in_channels=132, dilation=True)
        self.output_layer_7 = nn.Conv2d(64, 1, kernel_size=1)

        # meta selection
        self.selection_net = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # nn.Linear(256, 3),
            # nn.Softmax(),
        )
        self.se_fc = nn.Linear(256, 3)

        # Classifiers
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1, 1))

        self.cls_fc = nn.Linear(768, num_classes)

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
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        x_112 = self.avgpool2d(x)
        x_56 = self.avgpool2d(x_112)
        x_28 = self.avgpool2d(x_56)
        x_14 = self.avgpool2d(x_28)
        x_7 = self.avgpool2d(x_14)

        # Counting head

        htmap_28 = self.down_sample_28(p1)
        htmap_28 = self.backend_28(torch.cat([x_28, htmap_28], dim=1))
        htmap_28 = self.output_layer_28(htmap_28)
        htmap_28 = torch.sigmoid(htmap_28)

        htmap_14 = self.down_sample_14(p2)
        htmap_14 = self.backend_14(torch.cat([x_14, htmap_14, self.avgpool2d(htmap_28)], dim=1))
        htmap_14 = self.output_layer_14(htmap_14)
        htmap_14 = torch.sigmoid(htmap_14)

        htmap_7 = self.down_sample_7(p3)
        htmap_7 = self.backend_7(torch.cat([x_7, htmap_7, self.avgpool2d(htmap_14)], dim=1))
        htmap_7 = self.output_layer_7(htmap_7)
        htmap_7 = torch.sigmoid(htmap_7)

        # Attention head
        p4 = p4 * htmap_7
        p3 = p3 * htmap_14
        p2 = p2 * htmap_28

        # select head
        se_fea = torch.cat([p4, self.avgpool2d(p3), self.avgpool2d(self.avgpool2d(p2))], dim=1)
        se_fea = self.selection_net(se_fea)
        select_weight = self.se_fc(se_fea.view(se_fea.shape[0], -1))

        if epoch >= 4:
            p4 = p4 * select_weight[:, 0].view(select_weight.shape[0], 1, 1, 1)
            p3 = p3 * select_weight[:, 1].view(select_weight.shape[0], 1, 1, 1)
            p2 = p2 * select_weight[:, 2].view(select_weight.shape[0], 1, 1, 1)

        gap_p4 = self.gap(p4)
        gap_p3 = self.gap(p3)
        gap_p2 = self.gap(p2)

        final_fea = torch.cat([gap_p4, gap_p3, gap_p2], dim=1)
        final_logit = self.cls_fc(final_fea.view(final_fea.shape[0], -1))

        return final_logit, select_weight


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
            # nn.Linear(256, 3),
            # nn.Softmax(),
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
        # p4 = self.smooth1(p4)
        # p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)
        p1 = self.smooth4(p1)

        x_112 = self.avgpool2d(x)
        x_56 = self.avgpool2d(x_112)
        x_28 = self.avgpool2d(x_56)

        # Counting head
        ht_map_28 = self.down_sample(p1)
        ht_map_28 = self.backend_1(torch.cat([x_28, ht_map_28], dim=1))
        ht_map_28 = self.output_layer_1(ht_map_28)
        ht_map_28 = torch.sigmoid(ht_map_28)  # 28*28

        ht_map_56 = F.upsample(ht_map_28, size=(56, 56), mode='bilinear')
        ht_map_56 = self.backend_2(torch.cat([x_56, ht_map_56], dim=1))
        ht_map_56 = self.output_layer_2(ht_map_56)
        ht_map_56 = torch.sigmoid(ht_map_56)  # 56*56

        # # Class head
        # gap_p4 = self.gap(p4)
        # gap_p3 = self.gap(p3)
        # gap_p2 = self.gap(p2)
        # p4_logit = self.fc_1(gap_p4.view(gap_p4.shape[0], -1))
        # p3_logit = self.fc_1(gap_p3.view(gap_p3.shape[0], -1))
        # p2_logit = self.fc_1(gap_p2.view(gap_p2.shape[0], -1))

        # Attention class head
        ht_map_14 = self.avgpool2d(ht_map_28)
        ht_map_7 = self.avgpool2d(ht_map_14)

        p4 = p4 * ht_map_7
        p3 = p3 * ht_map_14
        p2 = p2 * ht_map_28

        # select head
        se_fea = torch.cat([p4, self.avgpool2d(p3), self.avgpool2d(self.avgpool2d(p2))], dim=1)
        se_fea = self.selection_net(se_fea)
        select_weight = torch.softmax(self.se_fc(se_fea.view(se_fea.shape[0], -1)), dim=1)

        # if epoch >= 4:
        #     p4 = p4 * select_weight[:, 0].view(select_weight.shape[0], 1, 1, 1)
        #     p3 = p3 * select_weight[:, 1].view(select_weight.shape[0], 1, 1, 1)
        #     p2 = p2 * select_weight[:, 2].view(select_weight.shape[0], 1, 1, 1)

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


class My_Counting_Net_Insight_Gated(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(My_Counting_Net_Insight_Gated, self).__init__()

        # self.resnet = models.resnet34(pretrained)  # resnst 34
        # self.resnet = models.resnet50(pretrained)  # resnst 50
        self.resnet = models.resnet101(pretrained)  # resnst 101
        # self.resnet = models.resnet152(pretrained)  # resnst 152

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
        p1 = self.smooth4(p1)

        x_112 = self.avgpool2d(x)
        x_56 = self.avgpool2d(x_112)
        x_28 = self.avgpool2d(x_56)

        # Counting head
        ht_map_1 = self.down_sample(p1)
        ht_map_1 = self.backend_1(torch.cat([x_28, ht_map_1], dim=1))
        ht_map_1 = self.output_layer_1(ht_map_1)
        ht_map_1 = torch.sigmoid(ht_map_1)

        ht_map_2 = F.upsample(ht_map_1, size=(p1.shape[2], p1.shape[3]), mode='bilinear')
        ht_map_2 = self.backend_2(torch.cat([x_56, ht_map_2], dim=1))
        ht_map_2 = self.output_layer_2(ht_map_2)
        ht_map_2 = torch.sigmoid(ht_map_2)

        # Attention class head
        ht_map_28 = self.avgpool2d(ht_map_2)
        ht_map_14 = self.avgpool2d(ht_map_28)
        ht_map_7 = self.avgpool2d(ht_map_14)

        p4 = p4 * ht_map_7
        p3 = p3 * ht_map_14
        p2 = p2 * ht_map_28

        # # select head
        # ga_fea = torch.cat([p4, self.avgpool2d(p3), self.avgpool2d(self.avgpool2d(p2))], dim=1)
        # ga_fea = self.gated_net(ga_fea)
        # gated_weight = torch.softmax(self.ga_fc(ga_fea.view(ga_fea.shape[0], -1)), dim=1)

        gap_p4 = self.gap(p4)
        gap_p3 = self.gap(p3)
        gap_p2 = self.gap(p2)

        p4_logit = self.fc_1(gap_p4.view(gap_p4.shape[0], -1))
        p3_logit = self.fc_2(gap_p3.view(gap_p3.shape[0], -1))
        p2_logit = self.fc_3(gap_p2.view(gap_p2.shape[0], -1))

        # p4_logit = p4_logit * gated_weight[:, 0].view(gated_weight.shape[0], 1)
        # p3_logit = p3_logit * gated_weight[:, 1].view(gated_weight.shape[0], 1)
        # p2_logit = p2_logit * gated_weight[:, 2].view(gated_weight.shape[0], 1)

        final_logit = p4_logit + p3_logit + p2_logit

        return final_logit
