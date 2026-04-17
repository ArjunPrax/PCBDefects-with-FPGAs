"""
threshold_sweep.py
------------------
Sweeps the defect classification threshold and reports precision/recall/F1
at each value.  At threshold t, a sample is classified as 'defect' if
softmax_prob(defect) >= t.

Saves:
  evaluation/threshold_sweep.json   -- full table
  evaluation/threshold_sweep.png    -- precision/recall/F1 curve

Usage:
    python scripts/threshold_sweep.py \\
        --checkpoint /path/to/best_model.pth \\
        --dataset datasets/deeppcb
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.model import create_model
from models.dataset import DefectDataset, get_transforms


# ─────────────────────────────────────────────────────────────
# Collect softmax probabilities on the val set
# ─────────────────────────────────────────────────────────────

def collect_probs(checkpoint_path, dataset_root, input_size, batch_size):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    num_classes = len(ckpt.get('class_to_idx', {'defect': 0, 'normal': 1}))
    class_to_idx = ckpt.get('class_to_idx', {'defect': 0, 'normal': 1})
    defect_idx = class_to_idx['defect']   # index of the defect class

    model = create_model('classifier', num_classes=num_classes,
                         input_size=input_size, freeze_backbone=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()

    _, val_tf = get_transforms(input_size=input_size, augment=False)
    val_ds = DefectDataset(root=dataset_root, split='val', transform=val_tf)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs  = []   # prob of defect class for each sample
    all_labels = []

    print(f"Running inference on {len(val_ds)} val samples...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)           # (B, 2)
            all_probs.append(probs[:, defect_idx].cpu().numpy())
            all_labels.append(labels.numpy())

    probs_arr  = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    return probs_arr, labels_arr, defect_idx


# ─────────────────────────────────────────────────────────────
# Sweep thresholds
# ─────────────────────────────────────────────────────────────

def sweep(probs, labels, defect_idx, thresholds):
    rows = []
    normal_idx = 1 - defect_idx   # works for binary (defect=0, normal=1)

    for t in thresholds:
        # pred_defect[i] = True  →  we call sample i a defect
        pred_defect = probs >= t
        true_defect = labels == defect_idx

        tp = int(( pred_defect &  true_defect).sum())
        fp = int(( pred_defect & ~true_defect).sum())
        fn = int((~pred_defect &  true_defect).sum())
        tn = int((~pred_defect & ~true_defect).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / len(labels)

        rows.append({
            'threshold': round(float(t), 2),
            'defect_precision': round(precision, 4),
            'defect_recall':    round(recall,    4),
            'f1':               round(f1,        4),
            'accuracy':         round(accuracy,  4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })
    return rows


# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────

def plot(rows, out_path, default_threshold=0.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ts   = [r['threshold']        for r in rows]
    prec = [r['defect_precision'] for r in rows]
    rec  = [r['defect_recall']    for r in rows]
    f1s  = [r['f1']               for r in rows]
    acc  = [r['accuracy']         for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, prec, 'b-o',  markersize=4, label='Defect Precision')
    ax.plot(ts, rec,  'r-o',  markersize=4, label='Defect Recall')
    ax.plot(ts, f1s,  'g-o',  markersize=4, label='F1 Score')
    ax.plot(ts, acc,  'k--',  markersize=3, linewidth=1, label='Overall Accuracy')

    ax.axvline(x=default_threshold, color='grey', linestyle=':', linewidth=1.5,
               label=f'Default threshold ({default_threshold})')

    ax.set_xlabel('Classification Threshold (defect probability)')
    ax.set_ylabel('Score')
    ax.set_title('Defect Classification: Precision / Recall / F1 vs. Threshold')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset',    default='datasets/deeppcb')
    parser.add_argument('--input-size', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', default='evaluation')
    parser.add_argument('--recall-target', type=float, default=0.90,
                        help='Find threshold that hits this defect recall (default 0.90)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    probs, labels, defect_idx = collect_probs(
        args.checkpoint, args.dataset, args.input_size, args.batch_size)

    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    rows = sweep(probs, labels, defect_idx, thresholds)

    # ── Print table ────────────────────────────────────────────
    print()
    print(f"{'Thresh':>7}  {'Defect Prec':>11}  {'Defect Rec':>10}  {'F1':>6}  {'Accuracy':>8}")
    print("-" * 55)
    for r in rows:
        marker = " ◀ default" if r['threshold'] == 0.50 else ""
        print(f"  {r['threshold']:.2f}    {r['defect_precision']:.4f}        "
              f"{r['defect_recall']:.4f}    {r['f1']:.4f}    {r['accuracy']:.4f}{marker}")

    # ── Find threshold for recall target ───────────────────────
    print()
    target = args.recall_target
    candidates = [r for r in rows if r['defect_recall'] >= target]
    if candidates:
        # Among those that meet recall target, pick highest F1
        best = max(candidates, key=lambda r: r['f1'])
        print(f"Best threshold for defect recall ≥ {target*100:.0f}%:")
        print(f"  threshold        = {best['threshold']}")
        print(f"  defect recall    = {best['defect_recall']*100:.2f}%")
        print(f"  defect precision = {best['defect_precision']*100:.2f}%")
        print(f"  F1               = {best['f1']:.4f}")
        print(f"  accuracy         = {best['accuracy']*100:.2f}%")
    else:
        best = None
        print(f"No threshold achieves ≥{target*100:.0f}% defect recall in this sweep.")

    # ── Save JSON ──────────────────────────────────────────────
    out = {
        'default_threshold': 0.5,
        'default_metrics': next(r for r in rows if r['threshold'] == 0.50),
        'recall_target': target,
        'best_for_recall_target': best,
        'sweep': rows,
    }
    json_path = os.path.join(args.output_dir, 'threshold_sweep.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nFull table saved to {json_path}")

    # ── Plot ───────────────────────────────────────────────────
    png_path = os.path.join(args.output_dir, 'threshold_sweep.png')
    plot(rows, png_path)


if __name__ == '__main__':
    main()
