"""
generate_figures.py
-------------------
Generates all figures needed for the paper:

  report/figures/confusion_matrix.png   — PCB defect classification CM
  report/figures/threshold_sweep.png    — Precision/recall/F1 vs threshold
  report/figures/architecture.png       — PS-PL system block diagram

Usage:
    python scripts/generate_figures.py
"""

import json
import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

FIGURES_DIR  = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
EVAL_DIR     = os.path.join(os.path.dirname(__file__), '..', 'evaluation')
THRESHOLD_PNG = os.path.join(EVAL_DIR, 'threshold_sweep.png')
METRICS_JSON  = os.path.join(EVAL_DIR, 'output', 'evaluation_metrics.json')
SWEEP_JSON    = os.path.join(EVAL_DIR, 'threshold_sweep.json')


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Figures directory: {os.path.abspath(FIGURES_DIR)}")


# ─────────────────────────────────────────────────────────────
# Figure 1: Confusion Matrix
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix():
    print("\n[1/3] Confusion matrix ...")

    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    y_true = np.array(metrics['labels'])
    y_pred = np.array(metrics['predictions'])

    # class 0 = defect, class 1 = normal (from checkpoint class_to_idx)
    class_names = ['Defect', 'Normal']

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Normalised values for colour, raw counts as text
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=11)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                    ha='center', va='center', fontsize=10,
                    color='white' if cm_norm[i, j] > thresh else 'black')

    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_title('PCB Defect Classification\nConfusion Matrix (val, n=1400)', fontsize=11)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 2: Threshold sweep (copy existing PNG + regenerate cleaner version)
# ─────────────────────────────────────────────────────────────

def plot_threshold_sweep():
    print("\n[2/3] Threshold sweep ...")

    with open(SWEEP_JSON) as f:
        data = json.load(f)

    rows = data['sweep']
    ts   = [r['threshold']        for r in rows]
    prec = [r['defect_precision'] for r in rows]
    rec  = [r['defect_recall']    for r in rows]
    f1s  = [r['f1']               for r in rows]
    acc  = [r['accuracy']         for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ts, prec, 'b-o',  markersize=5, linewidth=1.8, label='Defect Precision')
    ax.plot(ts, rec,  'r-o',  markersize=5, linewidth=1.8, label='Defect Recall')
    ax.plot(ts, f1s,  'g-o',  markersize=5, linewidth=1.8, label='F1 Score')
    ax.plot(ts, acc,  'k--',  markersize=3, linewidth=1.2, label='Overall Accuracy')

    # Annotate default threshold
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5,
               label='Default (t=0.50)')

    # Annotate best F1 threshold (0.20)
    best = max(rows, key=lambda r: r['f1'])
    ax.axvline(x=best['threshold'], color='green', linestyle='--', linewidth=1.5,
               label=f"Best F1 (t={best['threshold']:.2f})")

    # Annotate 90% recall threshold (0.10)
    recall90 = next((r for r in rows if r['defect_recall'] >= 0.90), None)
    if recall90:
        ax.axvline(x=recall90['threshold'], color='red', linestyle='--', linewidth=1.5,
                   label=f"90% recall (t={recall90['threshold']:.2f})")

    ax.set_xlabel('Classification Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('PCB Defect: Precision / Recall / F1 vs. Threshold', fontsize=11)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0.03, 0.97); ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'threshold_sweep.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 3: System architecture block diagram
# ─────────────────────────────────────────────────────────────

def plot_architecture():
    print("\n[3/3] System architecture diagram ...")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6)
    ax.axis('off')

    def box(ax, x, y, w, h, label, sublabel=None,
            facecolor='#d6eaf8', edgecolor='#2980b9', fontsize=10):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=facecolor, edgecolor=edgecolor,
                              linewidth=1.5)
        ax.add_patch(rect)
        ty = y + h/2 + (0.15 if sublabel else 0)
        ax.text(x + w/2, ty, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='#1a252f')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha='center', va='center',
                    fontsize=8, color='#555555')

    def arrow(ax, x1, y1, x2, y2, label='', color='#2c3e50'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.8, connectionstyle='arc3,rad=0'))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.15, label, ha='center', va='bottom',
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    # ── Title ────────────────────────────────────────────────
    ax.text(5.5, 5.65, 'FPGA-Accelerated Inference System — Xilinx ZCU104',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # ── Outer boundary boxes ─────────────────────────────────
    # PS (ARM) region
    ps_rect = FancyBboxPatch((0.2, 0.3), 4.5, 4.8,
                             boxstyle="round,pad=0.1",
                             facecolor='#eaf4fb', edgecolor='#2980b9',
                             linewidth=2, linestyle='--')
    ax.add_patch(ps_rect)
    ax.text(2.45, 5.0, 'Processing System (PS) — ARM Cortex-A53',
            ha='center', fontsize=9, color='#2980b9', fontstyle='italic')

    # PL (FPGA) region
    pl_rect = FancyBboxPatch((5.3, 0.3), 5.2, 4.8,
                             boxstyle="round,pad=0.1",
                             facecolor='#fef9e7', edgecolor='#e67e22',
                             linewidth=2, linestyle='--')
    ax.add_patch(pl_rect)
    ax.text(7.9, 5.0, 'Programmable Logic (PL) — Zynq UltraScale+ FPGA',
            ha='center', fontsize=9, color='#e67e22', fontstyle='italic')

    # ── PS blocks ────────────────────────────────────────────
    box(ax, 0.4, 3.4, 2.0, 0.9, 'Input Image',  '96×96 RGB',
        facecolor='#d5f5e3', edgecolor='#27ae60')
    box(ax, 0.4, 2.2, 2.0, 0.9, 'Preprocess',   'Resize + Normalize\n+ INT8 Quantize',
        facecolor='#d6eaf8', edgecolor='#2980b9')
    box(ax, 0.4, 1.0, 2.0, 0.9, 'Post-process', 'Softmax + Argmax\nThreshold Apply',
        facecolor='#d6eaf8', edgecolor='#2980b9')
    box(ax, 2.7, 1.0, 1.8, 0.9, 'Output',       'Defect / Normal',
        facecolor='#d5f5e3', edgecolor='#27ae60')

    # Python PYNQ driver
    box(ax, 0.4, 0.1, 4.1, 0.7, 'PYNQ Python Driver  (inference_driver.py)',
        facecolor='#f9ebea', edgecolor='#c0392b', fontsize=9)

    # ── AXI bus (middle) ─────────────────────────────────────
    # AXI-Lite control
    axi_lite = FancyBboxPatch((4.75, 2.8), 0.55, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#f5cba7', edgecolor='#e67e22', linewidth=1.5)
    ax.add_patch(axi_lite)
    ax.text(5.025, 3.55, 'AXI\nLite', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#784212')

    # AXI-Stream data
    axi_stream = FancyBboxPatch((4.75, 1.1), 0.55, 1.4,
                                boxstyle="round,pad=0.05",
                                facecolor='#f5cba7', edgecolor='#e67e22', linewidth=1.5)
    ax.add_patch(axi_stream)
    ax.text(5.025, 1.8, 'AXI\nStream', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#784212')

    # ── PL blocks ─────────────────────────────────────────────
    box(ax, 5.5, 3.4, 2.0, 0.9, 'Conv3×3 Engine',
        'INT8 MAC\nLine-buffer pipeline',
        facecolor='#fdebd0', edgecolor='#e67e22')
    box(ax, 5.5, 2.2, 2.0, 0.9, 'ReLU Module',
        'max(0,x)\n1-cycle latency',
        facecolor='#fdebd0', edgecolor='#e67e22')
    box(ax, 5.5, 1.0, 2.0, 0.9, 'Weight BRAM',
        'Dual-port 4KB\nINT8 weights',
        facecolor='#fdebd0', edgecolor='#e67e22')

    # AXI wrapper
    box(ax, 7.8, 1.0, 2.5, 3.3, 'AXI Wrapper\n(axi_conv_wrapper.v)',
        'Ctrl regs 0x00–0x14\nStart / Done / Busy',
        facecolor='#fef9e7', edgecolor='#e67e22', fontsize=9)

    # ── Arrows ────────────────────────────────────────────────
    # PS internal flow
    arrow(ax, 1.4, 3.4, 1.4, 3.1,  color='#2980b9')   # image → preprocess
    arrow(ax, 1.4, 2.2, 1.4, 1.9,  color='#2980b9')   # preprocess → postprocess
    arrow(ax, 2.4, 1.45, 2.7, 1.45, color='#27ae60')  # postprocess → output

    # PS → AXI
    arrow(ax, 2.4, 2.65, 4.75, 3.3, 'INT8 image\npixels', color='#e67e22')
    arrow(ax, 2.4, 3.0, 4.75, 3.0, 'Control\n(start/reset)', color='#c0392b')

    # AXI → PL
    arrow(ax, 5.3, 3.3, 5.5, 3.7,  color='#e67e22')
    arrow(ax, 5.3, 2.95, 5.5, 2.65, color='#e67e22')
    arrow(ax, 5.3, 1.5, 5.5, 1.5,  color='#e67e22')

    # PL internal
    arrow(ax, 6.5, 3.4, 6.5, 3.1, color='#e67e22')    # conv → relu
    arrow(ax, 6.5, 2.2, 6.5, 1.9, color='#e67e22')    # relu → bram area

    # AXI ← PL (results back)
    arrow(ax, 7.8, 2.65, 5.3, 2.0, 'Results', color='#8e44ad')

    # AXI → PS (results)
    arrow(ax, 4.75, 1.8, 2.4, 1.8, 'INT16\naccumulator', color='#8e44ad')

    plt.tight_layout(pad=0.5)
    out = os.path.join(FIGURES_DIR, 'architecture.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ensure_figures_dir()
    plot_confusion_matrix()
    plot_threshold_sweep()
    plot_architecture()
    print("\nAll figures generated successfully.")
    print(f"Location: {os.path.abspath(FIGURES_DIR)}")
