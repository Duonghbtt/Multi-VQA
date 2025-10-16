# multimodal fusion layer
# Fusion layer kết hợp image + text embeddings (concat hoặc cross-attention)
import torch
import torchvision.models as models
import torch.nn as nn
class Fusion(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(img_dim + txt_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, img_feat, txt_feat):
        x = torch.cat([img_feat, txt_feat], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
