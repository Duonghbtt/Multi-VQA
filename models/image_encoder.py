# Xây dựng CNN để encode ảnh thành vector feature
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=2048, pretrained=True):
        super().__init__()
        # Dùng weights thay vì pretrained
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = resnet50(weights=weights)
        modules = list(resnet.children())[:-1]  # bỏ fully connected cuối
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.resnet(x)  # (B, 2048, 1, 1)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        return feat  # (B, output_dim)
