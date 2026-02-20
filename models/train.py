"""
Training script - transfer learning version.

Two-phase training:
  Phase 1 (epochs 1-20):  Backbone frozen, only train head. Fast convergence.
  Phase 2 (epochs 21-50): Unfreeze all, fine-tune entire network at low LR.
"""

import os
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import Counter

from model import create_model
from dataset import DefectDataset, get_transforms
from utils import AverageMeter, save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    losses     = AverageMeter()
    accuracies = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, targets in pbar:
        images  = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss    = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        acc = 100. * predicted.eq(targets).sum().item() / targets.size(0)
        accuracies.update(acc, images.size(0))
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{accuracies.avg:.2f}%'})

    return losses.avg, accuracies.avg


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    losses     = AverageMeter()
    accuracies = AverageMeter()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val  ]')
        for images, targets in pbar:
            images  = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss    = criterion(outputs, targets)
            losses.update(loss.item(), images.size(0))
            _, predicted = outputs.max(1)
            acc = 100. * predicted.eq(targets).sum().item() / targets.size(0)
            accuracies.update(acc, images.size(0))
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{accuracies.avg:.2f}%'})

    # Collapse detector
    pred_dist   = Counter(all_preds)
    num_classes = len(set(all_targets))
    if len(pred_dist) == 1:
        print(f"\n  ⚠️  CLASS COLLAPSE epoch {epoch}: only predicting class {list(pred_dist.keys())[0]}")
    elif len(pred_dist) < num_classes:
        print(f"\n  ⚠️  Partial collapse: classes {set(range(num_classes)) - set(pred_dist.keys())} never predicted")

    return losses.avg, accuracies.avg, all_preds, all_targets


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms(args.input_size)
    train_dataset    = DefectDataset(root=args.dataset, split='train', transform=train_tf)
    val_dataset      = DefectDataset(root=args.dataset, split='val',   transform=val_tf)

    class_weights = train_dataset.get_class_weights().to(device)
    print(f"Class weights: {[round(w,4) for w in class_weights.tolist()]}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = train_dataset.num_classes
    model = create_model(model_type=args.model_type,
                         num_classes=num_classes,
                         input_size=args.input_size,
                         dropout=args.dropout,
                         freeze_backbone=True)   # Phase 1: frozen backbone
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1 optimizer: only parameters with requires_grad=True (head + layer4)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    history          = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc         = 0.0
    patience_counter = 0
    phase            = 1

    print(f"\n{'='*60}")
    print(f"PHASE 1: Training head only (backbone frozen)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):

        # ── Switch to Phase 2 at halfway point ────────────────────────────────
        if epoch == args.phase2_epoch and phase == 1:
            phase = 2
            print(f"\n{'='*60}")
            print(f"PHASE 2: Fine-tuning entire network (low LR)")
            print(f"{'='*60}\n")
            model.unfreeze_all()
            # Reset optimizer with lower LR for fine-tuning
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr / 10,
                weight_decay=args.weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
            )
            patience_counter = 0  # Reset patience for phase 2

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs}  |  Phase {phase}  |  LR: {current_lr:.2e}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, epoch)

        scheduler.step(val_loss)

        writer.add_scalar('Loss/train',     train_loss, epoch)
        writer.add_scalar('Loss/val',       val_loss,   epoch)
        writer.add_scalar('Accuracy/train', train_acc,  epoch)
        writer.add_scalar('Accuracy/val',   val_acc,    epoch)
        writer.add_scalar('LR',             current_lr, epoch)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\n  Train → loss: {train_loss:.4f}  acc: {train_acc:.2f}%")
        print(f"  Val   → loss: {val_loss:.4f}  acc: {val_acc:.2f}%")

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
            print(f"  ✓ New best: {best_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        save_checkpoint({
            'epoch':        epoch,
            'state_dict':   model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'best_acc':     best_acc,
            'args':         vars(args),
            'class_to_idx': train_dataset.class_to_idx,
        }, is_best, args.output_dir)

        if args.patience > 0 and patience_counter >= args.patience and epoch >= args.phase2_epoch:
            print(f"\n  Early stopping after {args.patience} non-improving epochs in Phase 2.")
            break

    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\nDone. Best val accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str,   default='datasets/deeppcb')
    parser.add_argument('--input-size',   type=int,   default=96)
    parser.add_argument('--model-type',   type=str,   default='classifier')
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--workers',      type=int,   default=0)
    parser.add_argument('--patience',     type=int,   default=10)
    parser.add_argument('--phase2-epoch', type=int,   default=20,
                        help='Epoch to unfreeze backbone for fine-tuning')
    parser.add_argument('--output-dir',   type=str,   default='models/checkpoints')
    parser.add_argument('--resume',       type=str,   default=None)
    args = parser.parse_args()
    main(args)
