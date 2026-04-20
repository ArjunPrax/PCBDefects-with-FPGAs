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
                   label=f"90%-recall (t={recall90['threshold']:.2f})")

    # Annotate 93% recall threshold (0.05) — near-zero-miss operating point
    recall93 = next((r for r in rows if r['threshold'] == 0.05), None)
    if recall93:
        ax.axvline(x=recall93['threshold'], color='blue', linestyle='--', linewidth=1.5,
                   label=f"93%-recall (t={recall93['threshold']:.2f})")

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

    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── Colour palette — greyscale academic style ─────────────
    C = {
        'bg':        'white',
        'ps_fill':   '#f0f0f0',
        'ps_border': '#333333',
        'pl_fill':   '#e4e4e4',
        'pl_border': '#333333',
        'block_ps':  '#ffffff',
        'block_pl':  '#ffffff',
        'block_io':  '#ffffff',
        'block_drv': '#ffffff',
        'axi':       '#d0d0d0',
        'text':      '#111111',
        'subtext':   '#444444',
        'arrow_ps':  '#111111',
        'arrow_pl':  '#111111',
        'arrow_ctrl':'#111111',
        'arrow_data':'#111111',
        'border_ps': '#333333',
        'border_pl': '#333333',
        'border_io': '#333333',
        'border_drv':'#333333',
        'axi_border':'#333333',
    }

    def rect(x, y, w, h, fc, ec, lw=1.5, ls='-', alpha=1.0):
        ax.add_patch(plt.Rectangle((x, y), w, h,
                     facecolor=fc, edgecolor=ec,
                     linewidth=lw, linestyle=ls, alpha=alpha))

    def label(x, y, txt, fs=9, color=None, bold=False, ha='center', va='center'):
        ax.text(x, y, txt, ha=ha, va=va, fontsize=fs,
                color=color or C['text'],
                fontweight='bold' if bold else 'normal')

    def block(x, y, w, h, title, subtitle='', fc=None, ec=None):
        fc = fc or C['block_ps']
        ec = ec or C['border_ps']
        rect(x, y, w, h, fc, ec, lw=1.8)
        if subtitle:
            label(x + w/2, y + h/2 + 0.18, title, fs=8.5, bold=True)
            label(x + w/2, y + h/2 - 0.18, subtitle, fs=7.5, color=C['subtext'])
        else:
            label(x + w/2, y + h/2, title, fs=8.5, bold=True)

    def arrowh(x1, x2, y, txt='', color=None, above=True):
        color = color or C['arrow_ps']
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.6,
                                   mutation_scale=14))
        if txt:
            ty = y + (0.18 if above else -0.18)
            label((x1+x2)/2, ty, txt, fs=7, color=color)

    def arrowv(x, y1, y2, txt='', color=None, right=True):
        color = color or C['arrow_ps']
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.6,
                                   mutation_scale=14))
        if txt:
            tx = x + (0.22 if right else -0.22)
            label(tx, (y1+y2)/2, txt, fs=7, color=color, ha='left' if right else 'right')

    def arrowdiag(x1, y1, x2, y2, txt='', color=None):
        color = color or C['arrow_ps']
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.6,
                                   mutation_scale=14,
                                   connectionstyle='arc3,rad=-0.2'))
        if txt:
            label((x1+x2)/2 + 0.1, (y1+y2)/2 + 0.2, txt, fs=7, color=color)

    # ── Region backgrounds ────────────────────────────────────
    rect(0.3, 0.4, 5.4, 6.1, C['ps_fill'], C['ps_border'], lw=2, ls='--', alpha=0.9)
    rect(6.5, 0.4, 6.1, 6.1, C['pl_fill'], C['pl_border'], lw=2, ls='--', alpha=0.9)

    # Region labels
    label(3.0, 6.3, 'Processing System (PS)  —  ARM Cortex-A53 @ 1.2 GHz',
          fs=8.5, color=C['border_ps'], bold=True)
    label(9.55, 6.3, 'Programmable Logic (PL)  —  Zynq UltraScale+ XCZU7EV',
          fs=8.5, color=C['border_pl'], bold=True)

    # ── PS blocks ────────────────────────────────────────────
    block(0.6, 4.6, 2.4, 1.0, 'Input Image', '96×96 RGB',
          fc=C['block_io'], ec=C['border_io'])
    block(0.6, 3.2, 2.4, 1.0, 'Preprocess',
          'Resize · Normalize · INT8 Quant',
          fc=C['block_ps'], ec=C['border_ps'])
    block(0.6, 1.8, 2.4, 1.0, 'Post-process',
          'Softmax · Argmax · Threshold',
          fc=C['block_ps'], ec=C['border_ps'])
    block(3.3, 1.8, 2.0, 1.0, 'Output',
          'Defect / Normal',
          fc=C['block_io'], ec=C['border_io'])
    block(0.6, 0.55, 4.7, 0.85, 'PYNQ Python Driver  (inference_driver.py)',
          fc=C['block_drv'], ec=C['border_drv'])

    # ── AXI bus column ────────────────────────────────────────
    rect(5.85, 3.05, 0.6, 1.4, C['axi'], C['axi_border'], lw=1.8)
    label(6.15, 3.75, 'AXI\nLite', fs=7.5, color=C['text'], bold=True)

    rect(5.85, 1.5, 0.6, 1.3, C['axi'], C['axi_border'], lw=1.8)
    label(6.15, 2.15, 'AXI\nStream', fs=7.5, color=C['text'], bold=True)

    # ── PL blocks ─────────────────────────────────────────────
    block(6.8, 4.5, 2.6, 1.1, 'Conv3×3 Engine',
          'INT8 MAC · Line-buffer pipeline',
          fc=C['block_pl'], ec=C['border_pl'])
    block(6.8, 3.1, 2.6, 1.1, 'ReLU Module',
          'out = max(0, in)  ·  0-cycle latency',
          fc=C['block_pl'], ec=C['border_pl'])
    block(6.8, 1.7, 2.6, 1.1, 'Weight BRAM',
          'Dual-port · INT8 weights',
          fc=C['block_pl'], ec=C['border_pl'])
    block(9.8, 1.7, 2.5, 3.9, 'AXI Wrapper',
          'axi_conv_wrapper.v\n0x00 Control\n0x04 Status\n0x08–0x14 Config',
          fc=C['block_pl'], ec=C['border_pl'])

    # ── Arrows — PS internal ──────────────────────────────────
    arrowv(1.8, 4.6, 4.2, color=C['arrow_ps'])        # image → preprocess
    arrowv(1.8, 3.2, 2.8, color=C['arrow_ps'])        # preprocess → postprocess
    arrowh(3.0, 3.3, 2.3, color=C['arrow_data'])      # postprocess → output

    # ── Arrows — PS → AXI ────────────────────────────────────
    arrowh(3.0, 5.85, 3.6, 'INT8 pixels', color=C['arrow_data'])
    arrowh(3.0, 5.85, 3.15, 'ctrl (start/reset)', color=C['arrow_ctrl'])

    # ── Arrows — AXI → PL ────────────────────────────────────
    arrowh(6.45, 6.8, 5.05, color=C['arrow_data'])    # → Conv3×3
    arrowh(6.45, 6.8, 3.65, color=C['arrow_ctrl'])    # → ReLU
    arrowh(6.45, 6.8, 2.25, color=C['arrow_data'])    # → BRAM

    # ── Arrows — PL internal ─────────────────────────────────
    arrowv(8.1, 4.5, 4.2, color=C['arrow_pl'])        # conv → relu
    arrowv(8.1, 3.1, 2.8, color=C['arrow_pl'])        # relu → bram

    # ── Arrow — results back to PS ───────────────────────────
    arrowdiag(6.8, 2.25, 5.85, 2.0, 'INT16 accum', color=C['arrow_pl'])
    arrowh(5.85, 3.0, 2.1, 'result', color=C['arrow_pl'], above=False)

    # ── Title ─────────────────────────────────────────────────
    label(6.5, 6.75,
          'FPGA-Accelerated Inference System — Xilinx ZCU104',
          fs=11, bold=True, color='#ffffff')

    plt.tight_layout(pad=0.3)
    out = os.path.join(FIGURES_DIR, 'architecture.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
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
