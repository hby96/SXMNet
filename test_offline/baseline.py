import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


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


class My_Counting_Net_Insight_Gated(nn.Module):
    def __init__(self, num_classes=1000, backbone='resnet50', pretrained=True):
        super(My_Counting_Net_Insight_Gated, self).__init__()

        self.backbone = backbone
        self.resnet = eval("models.{}".format(backbone))(pretrained)

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

        if self.backbone == 'resnet101':
            gap_p4 = self.gap(p4)
            gap_p3 = self.gap(p3)
            gap_p2 = self.gap(p2)

            p4_logit = self.fc_1(gap_p4.view(gap_p4.shape[0], -1))
            p3_logit = self.fc_2(gap_p3.view(gap_p3.shape[0], -1))
            p2_logit = self.fc_3(gap_p2.view(gap_p2.shape[0], -1))
        else:
            # select head
            ga_fea = torch.cat([p4, self.avgpool2d(p3), self.avgpool2d(self.avgpool2d(p2))], dim=1)
            ga_fea = self.gated_net(ga_fea)
            gated_weight = torch.softmax(self.ga_fc(ga_fea.view(ga_fea.shape[0], -1)), dim=1)

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
