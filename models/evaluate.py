"""
Model evaluation script
Computes accuracy, mAP, confusion matrix, and other metrics
"""

import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import create_model
from dataset import DefectDataset, get_transforms
from utils import (
    plot_confusion_matrix,
    print_classification_report,
    count_model_parameters,
    calculate_model_size,
    visualize_batch
)


def evaluate_classifier(model, dataloader, device, class_names):
    """
    Evaluate classification model
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = 100.0 * np.mean(all_preds == all_labels)
    
    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = 100.0 * np.mean(all_preds[mask] == all_labels[mask])
            per_class_acc[class_name] = class_acc
    
    metrics = {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'num_samples': len(all_labels),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }
    
    return metrics, all_preds, all_labels


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    _, val_transform = get_transforms(args.input_size)
    
    test_dataset = DefectDataset(
        root=args.dataset,
        split=args.split,
        transform=val_transform
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {test_dataset.num_classes}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    
    # Get class names
    class_names = list(test_dataset.class_to_idx.keys())
    print(f"Classes: {class_names}")
    
    # Create model
    model = create_model(
        model_type=args.model_type,
        num_classes=test_dataset.num_classes,
        input_size=args.input_size
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    # Model info
    print("\nModel Information:")
    print("=" * 60)
    count_model_parameters(model)
    calculate_model_size(model)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    metrics, all_preds, all_labels = evaluate_classifier(
        model, test_loader, device, class_names
    )
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")
    print("\nPer-Class Accuracy:")
    for class_name, acc in metrics['per_class_accuracy'].items():
        print(f"  {class_name}: {acc:.2f}%")
    
    # Classification report
    print_classification_report(all_labels, all_preds, class_names)
    
    # Confusion matrix
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)
        print(f"\nConfusion matrix saved to: {cm_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate defect detection model')
    
    # Data
    parser.add_argument('--dataset', type=str, default='datasets/deeppcb',
                        help='Path to dataset')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--input-size', type=int, default=96,
                        help='Input image size')
    
    # Model
    parser.add_argument('--model-type', type=str, default='classifier',
                        choices=['classifier', 'detector'],
                        help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Evaluation
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='evaluation',
                        help='Output directory')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots')
    
    args = parser.parse_args()
    main(args)
