"""
Utility functions for training and evaluation
"""

import os
import shutil
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """Save checkpoint to disk"""
    filepath = os.path.join(output_dir, 'checkpoints', filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filepath):
    """Load checkpoint from disk"""
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        return checkpoint
    else:
        raise FileNotFoundError(f"No checkpoint found at {filepath}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report (robust to missing classes in a split)"""
    labels = list(range(len(class_names)))  # e.g. [0,1]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )
    print("\nClassification Report:")
    print("=" * 60)
    print(report)



def count_model_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params


def calculate_model_size(model, save_path=None):
    """Calculate model size in MB"""
    if save_path:
        torch.save(model.state_dict(), save_path)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
    else:
        # Temporary save
        temp_path = 'temp_model.pth'
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
    
    print(f"Model size: {size_mb:.2f} MB")
    return size_mb


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def visualize_batch(images, labels, predictions=None, class_names=None, num_images=8):
    """
    Visualize a batch of images with labels and predictions
    
    Args:
        images: Batch of images (B, C, H, W)
        labels: Ground truth labels
        predictions: Predicted labels (optional)
        class_names: List of class names (optional)
        num_images: Number of images to display
    """
    num_images = min(num_images, len(images))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = denormalize(images[i]).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        label_name = class_names[labels[i]] if class_names else str(labels[i])
        title = f"GT: {label_name}"
        
        if predictions is not None:
            pred_name = class_names[predictions[i]] if class_names else str(predictions[i])
            color = 'green' if predictions[i] == labels[i] else 'red'
            title += f"\nPred: {pred_name}"
            axes[i].set_title(title, color=color)
        else:
            axes[i].set_title(title)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
