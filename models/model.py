"""
Lightweight CNN Model for Industrial Defect Detection
Optimized for FPGA deployment on Xilinx ZCU104

Model characteristics:
- Input: 96x96 or 224x224 grayscale/RGB images
- Parameters: < 2M
- Designed for quantization-friendly operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightDefectCNN(nn.Module):
    """
    Lightweight CNN for binary classification (defect/no-defect)
    or multi-class defect type classification.
    
    Architecture designed for FPGA acceleration:
    - 3x3 convolutions (easy to accelerate)
    - Minimal pooling (use strided conv instead)
    - Quantization-aware design
    """
    
    def __init__(self, num_classes=2, input_channels=3, input_size=96):
        super(LightweightDefectCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Block 1: 96x96 -> 48x48
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Block 2: 48x48 -> 24x24
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Block 3: 24x24 -> 12x12
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Block 4: 12x12 -> 6x6
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Compute feature map size
        feature_size = input_size // 16  # After 4 stride-2 convs
        fc_input_size = 128 * feature_size * feature_size
        
        # Classifier
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyYOLOv5(nn.Module):
    """
    Ultra-lightweight object detection model inspired by YOLOv5n
    Suitable for simple defect localization
    """
    
    def __init__(self, num_classes=1, input_size=224):
        super(TinyYOLOv5, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)  # 224->112
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)  # 112->56
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)  # 56->28
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)  # 28->14
        
        # Detection head (simplified)
        # Output: [batch, (5 + num_classes), grid_h, grid_w]
        # 5 = (x, y, w, h, confidence)
        self.detect = nn.Conv2d(128, (5 + num_classes) * 3, 1)  # 3 anchors
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        detection = self.detect(x)
        
        return detection


def create_model(model_type='classifier', num_classes=2, input_size=96):
    """
    Factory function to create models
    
    Args:
        model_type: 'classifier' or 'detector'
        num_classes: Number of classes (for classifier or detector)
        input_size: Input image size (96, 128, 224, etc.)
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'classifier':
        model = LightweightDefectCNN(num_classes=num_classes, input_size=input_size)
    elif model_type == 'detector':
        model = TinyYOLOv5(num_classes=num_classes, input_size=input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model created: {model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("=" * 50)
    print("Testing Lightweight CNN Classifier")
    print("=" * 50)
    model = create_model('classifier', num_classes=5, input_size=96)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 96, 96)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {model.count_parameters():,}")
    
    print("\n" + "=" * 50)
    print("Testing Tiny YOLO Detector")
    print("=" * 50)
    detector = create_model('detector', num_classes=1, input_size=224)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    detection = detector(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Detection output shape: {detection.shape}")
