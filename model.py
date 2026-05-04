import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)