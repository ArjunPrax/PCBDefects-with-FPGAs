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
from collections import Counter


class DefectDataset(Dataset):
    """
    Generic defect detection dataset.

    Expects folder structure:
        dataset_root/
            train/
                defect/
                    img1.jpg ...
                normal/
                    img1.jpg ...
            val/
                defect/ ...
                normal/ ...

    Args:
        root:        Path to dataset root folder
        split:       'train' or 'val'
        transform:   torchvision transforms to apply
        dummy_size:  If > 0 AND root does not exist, fall back to this many
                     dummy samples (useful for CI/unit-test only).
                     Set to 0 (default) to raise an error when data is missing.
    """

    def __init__(self, root, split='train', transform=None, dummy_size=0):
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}

        split_dir = os.path.join(root, split)

        if os.path.exists(split_dir):
            # ── Real data path ──────────────────────────────────────────────
            classes = sorted([
                d for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ])

            if not classes:
                raise RuntimeError(
                    f"No class sub-folders found in {split_dir}.\n"
                    f"Expected structure: {split_dir}/<class_name>/<image files>"
                )

            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

            for class_name, class_idx in self.class_to_idx.items():
                class_dir = os.path.join(split_dir, class_name)
                found = 0
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
                        found += 1

                if found == 0:
                    print(f"  ⚠️  Warning: no images found in {class_dir}")

            if not self.samples:
                raise RuntimeError(
                    f"No images found under {split_dir}. "
                    f"Check that your images have extensions: .png .jpg .jpeg .bmp"
                )

            # ── Class balance report ─────────────────────────────────────────
            self._report_balance()

        elif dummy_size > 0:
            # ── Dummy data (explicit opt-in only) ────────────────────────────
            print(f"⚠️  WARNING: {split_dir} not found.")
            print(f"   Running with {dummy_size} DUMMY samples (random images).")
            print(f"   Results will be MEANINGLESS. Add real data before evaluating.")
            self.class_to_idx = {'defect': 0, 'normal': 1}
            for i in range(dummy_size):
                self.samples.append((f'__dummy___{i}', i % 2))
        else:
            # ── Hard error ───────────────────────────────────────────────────
            raise FileNotFoundError(
                f"\n❌ Dataset split not found: {split_dir}\n\n"
                f"Expected structure:\n"
                f"  {root}/\n"
                f"  ├── train/\n"
                f"  │   ├── defect/   ← defect images here\n"
                f"  │   └── normal/   ← normal images here\n"
                f"  └── val/\n"
                f"      ├── defect/\n"
                f"      └── normal/\n\n"
                f"To use dummy data for unit tests, pass dummy_size=100."
            )

        self.num_classes = len(self.class_to_idx)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _report_balance(self):
        """Print class distribution so imbalance is always visible."""
        counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        idx_to_cls = {v: k for k, v in self.class_to_idx.items()}

        print(f"  [{self.split}] {total} samples loaded:")
        for idx in sorted(counts):
            n = counts[idx]
            pct = 100.0 * n / total
            flag = " ⚠️ IMBALANCED" if pct < 20 or pct > 80 else ""
            print(f"    {idx_to_cls[idx]} (class {idx}): {n} ({pct:.1f}%){flag}")

    def get_class_weights(self):
        """
        Compute inverse-frequency class weights for CrossEntropyLoss.

        Returns:
            torch.Tensor of shape (num_classes,)
        """
        counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = []
        for idx in range(self.num_classes):
            n = counts.get(idx, 1)           # avoid divide-by-zero
            weights.append(total / (self.num_classes * n))
        return torch.tensor(weights, dtype=torch.float32)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        if img_path.startswith('__dummy__'):
            # Dummy image path (unit-test mode only)
            image = Image.fromarray(
                np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            )
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(input_size=96, augment=True):
    """
    Get train and validation transforms.

    Args:
        input_size: Target image size (square)
        augment:    Apply augmentation to train transform

    Returns:
        (train_transform, val_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


if __name__ == '__main__':
    print("Testing DefectDataset...")
    train_tf, val_tf = get_transforms(input_size=96)

    # Will raise FileNotFoundError if data not present — that's intentional
    try:
        dataset = DefectDataset(root='datasets/deeppcb', split='train',
                                transform=train_tf)
        print(f"Class mapping: {dataset.class_to_idx}")
        print(f"Recommended class weights: {dataset.get_class_weights()}")

        img, label = dataset[0]
        print(f"Sample image shape: {img.shape}, label: {label}")
    except FileNotFoundError as e:
        print(e)
