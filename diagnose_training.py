"""
Diagnostic script - run this FIRST before retraining.
Identifies why model predicts all class 0.

Usage:
    python diagnose_training.py --dataset datasets/deeppcb
    python diagnose_training.py --checkpoint models/checkpoints/best_model.pth --dataset datasets/deeppcb
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.dataset import DefectDataset, get_transforms
from models.model import create_model


def check_dataset(dataset_path):
    """
    CHECK 1: Is the real dataset loading, or is dummy data being used?
    CHECK 2: Is the dataset balanced?
    CHECK 3: Are images valid?
    """
    print("\n" + "=" * 60)
    print("CHECK 1 & 2: Dataset Loading + Class Balance")
    print("=" * 60)

    _, val_transform = get_transforms(96)

    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_path, split)
        print(f"\n--- Split: {split} ---")

        if not os.path.exists(split_dir):
            print(f"  ❌ PROBLEM: {split_dir} does not exist!")
            print(f"     → Model is training on DUMMY DATA (random images).")
            print(f"     → Fix: create the folder structure:")
            print(f"       {dataset_path}/")
            print(f"       ├── train/")
            print(f"       │   ├── defect/   ← put defect images here")
            print(f"       │   └── normal/   ← put normal images here")
            print(f"       └── val/")
            print(f"           ├── defect/")
            print(f"           └── normal/")
            continue

        dataset = DefectDataset(root=dataset_path, split=split, transform=val_transform)
        print(f"  ✓ Directory exists")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Classes: {dataset.class_to_idx}")

        # Count per class
        label_counts = Counter(label for _, label in dataset.samples)
        print(f"  Samples per class:")
        for cls_name, cls_idx in dataset.class_to_idx.items():
            count = label_counts.get(cls_idx, 0)
            pct = 100.0 * count / len(dataset) if len(dataset) > 0 else 0
            status = "✓" if 30 < pct < 70 else "⚠️ IMBALANCED"
            print(f"    {cls_name} (idx={cls_idx}): {count} samples ({pct:.1f}%) {status}")

        # Check a few images are valid
        print(f"  Checking first 5 images...")
        bad_images = []
        for i in range(min(5, len(dataset))):
            img_path, label = dataset.samples[i]
            if not os.path.exists(img_path):
                bad_images.append(img_path)
            else:
                try:
                    img, lbl = dataset[i]
                    print(f"    [{i}] shape={img.shape}, label={lbl}, path OK")
                except Exception as e:
                    bad_images.append(f"{img_path}: {e}")

        if bad_images:
            print(f"  ❌ Bad images found: {bad_images}")
        else:
            print(f"  ✓ All sampled images loaded successfully")


def check_model_outputs(checkpoint_path, dataset_path):
    """
    CHECK 3: What are the raw model outputs (logits)?
    If logits are always higher for class 0, the model has collapsed.
    """
    print("\n" + "=" * 60)
    print("CHECK 3: Model Output Distribution")
    print("=" * 60)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("  No checkpoint provided or found — skipping model output check.")
        return

    device = torch.device('cpu')  # Use CPU for diagnosis
    checkpoint = torch.load(checkpoint_path, map_location=device)

    _, val_transform = get_transforms(96)
    dataset = DefectDataset(root=dataset_path, split='val', transform=val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    num_classes = dataset.num_classes
    model = create_model('classifier', num_classes=num_classes, input_size=96)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_logits.append(logits.numpy())
            all_probs.append(probs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_logits = np.concatenate(all_logits)
    all_probs = np.concatenate(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n  Logit statistics (per class):")
    for i in range(num_classes):
        logits_i = all_logits[:, i]
        print(f"    Class {i}: mean={logits_i.mean():.3f}, std={logits_i.std():.3f}, "
              f"min={logits_i.min():.3f}, max={logits_i.max():.3f}")

    print(f"\n  Predicted class distribution:")
    pred_counts = Counter(all_preds.tolist())
    for cls_idx, count in sorted(pred_counts.items()):
        print(f"    Class {cls_idx}: {count} predictions ({100.0*count/len(all_preds):.1f}%)")

    if len(pred_counts) == 1:
        collapsed_class = list(pred_counts.keys())[0]
        print(f"\n  ❌ CONFIRMED: Model collapsed — predicting ONLY class {collapsed_class}")
        print(f"\n  Diagnosis:")

        # Check logit gap
        mean_gap = all_logits[:, 0].mean() - all_logits[:, 1].mean()
        print(f"    Mean logit gap (class0 - class1): {mean_gap:.3f}")
        if abs(mean_gap) > 2.0:
            print(f"    → Logits are far apart — model is very confident in one class.")
            print(f"    → Likely cause: class imbalance or wrong class weights in training.")
    else:
        accuracy = 100.0 * (all_preds == all_labels).mean()
        print(f"\n  ✓ Model predicts both classes. Accuracy: {accuracy:.2f}%")

    print(f"\n  Confidence distribution:")
    max_probs = all_probs.max(axis=1)
    print(f"    Mean confidence: {max_probs.mean():.3f}")
    print(f"    Samples with >90% confidence: {(max_probs > 0.9).sum()}")
    print(f"    Samples with <60% confidence: {(max_probs < 0.6).sum()}")


def check_class_weights(dataset_path):
    """
    CHECK 4: What class weights SHOULD you use?
    Computes the correct inverse-frequency weights.
    """
    print("\n" + "=" * 60)
    print("CHECK 4: Recommended Class Weights")
    print("=" * 60)

    _, val_transform = get_transforms(96)
    dataset = DefectDataset(root=dataset_path, split='train', transform=val_transform)

    label_counts = Counter(label for _, label in dataset.samples)
    total = len(dataset.samples)

    print(f"\n  Training set class distribution:")
    weights = {}
    for cls_name, cls_idx in sorted(dataset.class_to_idx.items(), key=lambda x: x[1]):
        count = label_counts.get(cls_idx, 1)
        # Inverse frequency weighting
        weight = total / (len(dataset.class_to_idx) * count)
        weights[cls_idx] = weight
        print(f"    {cls_name} (idx={cls_idx}): {count} samples → weight={weight:.4f}")

    weight_list = [weights[i] for i in range(len(weights))]
    print(f"\n  → Use in train.py:")
    print(f"    class_weights = torch.tensor({[round(w, 4) for w in weight_list]})")
    print(f"\n  Current hardcoded weights in train.py: [1.0, 2.0]")

    if abs(weight_list[0] - 1.0) < 0.1 and abs(weight_list[1] - 2.0) < 0.1:
        print(f"  ✓ Current weights roughly match recommended weights")
    else:
        print(f"  ⚠️  Current weights DON'T match recommended — this may cause collapse")


def main():
    parser = argparse.ArgumentParser(description='Diagnose 50% accuracy problem')
    parser.add_argument('--dataset', type=str, default='datasets/deeppcb',
                        help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING DIAGNOSIS TOOL")
    print("=" * 60)
    print(f"Dataset path: {os.path.abspath(args.dataset)}")

    check_dataset(args.dataset)
    check_class_weights(args.dataset)

    if args.checkpoint:
        check_model_outputs(args.checkpoint, args.dataset)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. If dataset path is wrong:
   → Create the folder structure shown above and add your images

2. If dataset is severely imbalanced:
   → Use the recommended class_weights printed above in train.py

3. If model has collapsed (all same prediction):
   → Retrain with:
       a. Correct class weights (from Check 4 above)
       b. Lower learning rate: --lr 1e-4
       c. Add --check-dataset flag to verify data loads before training

4. Always verify before training:
   python diagnose_training.py --dataset datasets/deeppcb
""")


if __name__ == '__main__':
    main()