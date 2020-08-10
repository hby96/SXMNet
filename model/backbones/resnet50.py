import torch
from torch import nn
from torchvision import models

class ResNet50_Pre_Trained(nn.Module):
    def __init__(self, pre_trained, num_classes=1000):
        super(ResNet50_Pre_Trained, self).__init__()

        self.final_channel = 2048
        self.num_classes = num_classes

        self.resnet = models.resnet50(pre_trained)
        self.backbone = nn.Sequential(
            *list(self.resnet.children())[:-2] # output 7*7 feature map
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResNet50_Pre_Trained_5(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet50_Pre_Trained_5, self).__init__()

        self.final_channel = 2048
        self.num_classes = num_classes

        self.resnet = models.resnet50(False)
        self.backbone = nn.Sequential(
            *list(self.resnet.children())[:-2] # output 7*7 feature map
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x