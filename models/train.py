"""
Training script for defect detection model
Supports both classification and detection tasks
"""

import os
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import create_model
from dataset import DefectDataset, get_transforms
from utils import AverageMeter, save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        losses.update(loss.item(), images.size(0))
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = 100. * correct / targets.size(0)
        accuracies.update(acc, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
    
    return losses.avg, accuracies.avg


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Metrics
            losses.update(loss.item(), images.size(0))
            
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            acc = 100. * correct / targets.size(0)
            accuracies.update(acc, images.size(0))
            
            # Store predictions
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.2f}%'
            })
    
    return losses.avg, accuracies.avg, all_preds, all_targets


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Create datasets
    train_transform, val_transform = get_transforms(args.input_size)
    
    train_dataset = DefectDataset(
        root=args.dataset,
        split='train',
        transform=train_transform
    )
    
    val_dataset = DefectDataset(
        root=args.dataset,
        split='val',
        transform=val_transform
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Create model
    num_classes = train_dataset.num_classes
    model = create_model(
        model_type=args.model_type,
        num_classes=num_classes,
        input_size=args.input_size
    )
    model = model.to(device)
    
    # Loss and optimizer
    class_weights = torch.tensor([1.0, 1.0])
    if num_classes == 2:
        # heavier penalty for misclassifying NORMAL (label = 1)
        class_weights = torch.tensor([1.0, 2.0])

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(
            model, val_loader, criterion, device, epoch+1
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'args': vars(args)
        }, is_best, args.output_dir)
        
        print(f"  Best Val Acc: {best_acc:.2f}%")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    writer.close()
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train defect detection model')
    
    # Data
    parser.add_argument('--dataset', type=str, default='datasets/deeppcb',
                        help='Path to dataset')
    parser.add_argument('--input-size', type=int, default=96,
                        help='Input image size')
    
    # Model
    parser.add_argument('--model-type', type=str, default='classifier',
                        choices=['classifier', 'detector'],
                        help='Model type')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='models/checkpoints',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    main(args)
