"""
Dataset classes for defect detection
Supports both classification and detection tasks
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class DefectDataset(Dataset):
    """
    Generic defect detection dataset
    Assumes folder structure:
        dataset_root/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    ...
            val/
                ...
    """
    
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load samples
        self.samples = []
        self.class_to_idx = {}
        
        split_dir = os.path.join(root, split)
        
        if os.path.exists(split_dir):
            classes = sorted([d for d in os.listdir(split_dir) 
                            if os.path.isdir(os.path.join(split_dir, d))])
            
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            for class_name, class_idx in self.class_to_idx.items():
                class_dir = os.path.join(split_dir, class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
        else:
            # If split directory doesn't exist, create dummy data
            print(f"Warning: {split_dir} not found. Using dummy data.")
            self.class_to_idx = {'defect': 0, 'normal': 1}
            # Create some dummy entries
            for i in range(100):
                self.samples.append((f'dummy_{i}.jpg', i % 2))
        
        self.num_classes = len(self.class_to_idx)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image (or create dummy if path doesn't exist)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Dummy image for testing
            image = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size=96, augment=True):
    """
    Get train and validation transforms
    
    Args:
        input_size: Target image size
        augment: Whether to apply augmentation
    
    Returns:
        train_transform, val_transform
    """
    
    # Training transforms (with augmentation)
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


if __name__ == '__main__':
    # Test dataset
    print("Testing DefectDataset...")
    
    train_transform, val_transform = get_transforms(input_size=96)
    
    dataset = DefectDataset(
        root='datasets/deeppcb',
        split='train',
        transform=train_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class mapping: {dataset.class_to_idx}")
    
    # Test loading a sample
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Label: {label}")
