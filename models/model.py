"""
Lightweight CNN Model for Industrial Defect Detection
Uses pretrained ResNet18 backbone for transfer learning.

Why transfer learning fixes the 0.693 loss problem:
- 1700 samples is too small to train a CNN from scratch
- PCB defect images have subtle differences that need learned features
- ResNet18 pretrained on ImageNet already knows edges/textures/shapes
- We freeze most of the backbone and only train the small head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LightweightDefectCNN(nn.Module):
    """
    Transfer learning classifier using pretrained ResNet18 backbone.
    Only the head + last ResNet block are trainable by default.
    """

    def __init__(self, num_classes=2, input_channels=3, input_size=96,
                 dropout=0.3, freeze_backbone=True):
        super().__init__()

        self.input_size  = input_size
        self.num_classes = num_classes

        # Pretrained ResNet18, strip final FC layer
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output shape: (batch, 512, 1, 1)

        if freeze_backbone:
            # Freeze entire backbone first
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Then unfreeze layer4 (last block) for light fine-tuning
            for param in self.backbone[-3].parameters():
                param.requires_grad = True

        # Small classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(512, 128)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)           # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)      # (batch, 512)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def unfreeze_all(self):
        """Unfreeze entire backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone fully unfrozen.")

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class TinyYOLOv5(nn.Module):
    def __init__(self, num_classes=1, input_size=224, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.conv1  = nn.Conv2d(3,   16,  3, 2, 1)
        self.conv2  = nn.Conv2d(16,  32,  3, 2, 1)
        self.conv3  = nn.Conv2d(32,  64,  3, 2, 1)
        self.conv4  = nn.Conv2d(64,  128, 3, 2, 1)
        self.detect = nn.Conv2d(128, (5 + num_classes) * 3, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.detect(x)


def create_model(model_type='classifier', num_classes=2, input_size=96,
                 dropout=0.3, freeze_backbone=True):
    if model_type == 'classifier':
        model = LightweightDefectCNN(
            num_classes=num_classes,
            input_size=input_size,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
        total, trainable = model.count_parameters()
        print(f"Model      : ResNet18 + custom head (transfer learning)")
        print(f"Parameters : {total:,} total  |  {trainable:,} trainable")
        print(f"Backbone   : {'head + layer4 unfrozen' if freeze_backbone else 'fully unfrozen'}")
        print(f"Dropout    : {dropout}")
    elif model_type == 'detector':
        model = TinyYOLOv5(num_classes=num_classes, input_size=input_size)
        total = sum(p.numel() for p in model.parameters())
        print(f"Model: TinyYOLOv5 | Parameters: {total:,}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


if __name__ == "__main__":
    model = create_model('classifier', num_classes=2, input_size=96)
    x   = torch.randn(4, 3, 96, 96)
    out = model(x)
    print(f"Input {x.shape} -> Output {out.shape}")
