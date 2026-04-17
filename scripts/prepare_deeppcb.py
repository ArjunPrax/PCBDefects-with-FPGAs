"""
prepare_deeppcb.py
------------------
Converts the raw tangsanli5201/DeepPCB repository into the folder structure
expected by models/dataset.py:

    <dst>/
        train/
            defect/   ← *_test.jpg images (PCBs with visible defects)
            normal/   ← *_temp.jpg images (defect-free reference templates)
        val/
            defect/
            normal/

Raw DeepPCB layout:
    <src>/PCBData/
        groupXXXXX/
            XXXXX/
                XXXXXNNN_test.jpg   ← defect test image
                XXXXXNNN_temp.jpg   ← defect-free template image  (→ normal class)
            XXXXX_not/
                XXXXXNNN.txt        ← bounding-box annotations for test image
        trainval.txt    999 lines:  "groupXX/XX/XXNNN.jpg  groupXX/XX_not/XXNNN.txt"
        test.txt        499 lines:  same format

Classification rule:
    *_test.jpg  →  defect   (has visible PCB defects; annotated in _not/*.txt)
    *_temp.jpg  →  normal   (defect-free reference template)

Split strategy:
    - trainval.txt entries → 80 % train / 20 % val  (seeded shuffle)
    - test.txt entries     → appended to val
    Both the _test and _temp images for each entry are copied to the
    same split so images from the same PCB board stay together.

Usage:
    # 1. Clone the dataset (one-time, ~200 MB):
    #    git clone https://github.com/tangsanli5201/DeepPCB /tmp/DeepPCB_raw

    # 2. Run this script:
    python scripts/prepare_deeppcb.py \\
        --src /tmp/DeepPCB_raw \\
        --dst datasets/deeppcb
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_listfile(listfile: Path) -> list[Path]:
    """
    Parse trainval.txt / test.txt and return the image base paths.

    Each line looks like:
        group00041/00041/00041000.jpg  group00041/00041_not/00041000.txt

    We only need the image half (first token). Strip the .jpg suffix to get
    the base so we can append _test.jpg and _temp.jpg ourselves.
    """
    entries = []
    with listfile.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_rel = line.split()[0]           # e.g. group00041/00041/00041000.jpg
            base = img_rel[:-4]                 # strip .jpg  → group00041/00041/00041000
            entries.append(Path(base))
    return entries


def resolve_pair(pcb_data: Path, base: Path) -> tuple[Path | None, Path | None]:
    """
    Given a base path like group00041/00041/00041000, return
    (test_img_path, temp_img_path) under pcb_data.
    Returns None for a file if it doesn't exist on disk.
    """
    test_img = pcb_data / (str(base) + '_test.jpg')
    temp_img = pcb_data / (str(base) + '_temp.jpg')
    return (
        test_img if test_img.exists() else None,
        temp_img if temp_img.exists() else None,
    )


def copy_img(src: Path, dst_dir: Path, prefix: str = '') -> None:
    """Copy src into dst_dir, prepending prefix to avoid name collisions."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    unique_name = prefix + src.name
    shutil.copy2(src, dst_dir / unique_name)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert raw DeepPCB repo → models/dataset.py folder structure'
    )
    parser.add_argument('--src', required=True,
                        help='Path to cloned tangsanli5201/DeepPCB repo')
    parser.add_argument('--dst', default='datasets/deeppcb',
                        help='Destination root (default: datasets/deeppcb)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of trainval entries held out for val '
                             '(default: 0.2).  test.txt entries always go to val.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the train/val split (default: 42)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete and recreate --dst if it already exists')
    args = parser.parse_args()

    src      = Path(args.src).resolve()
    dst      = Path(args.dst).resolve()
    pcb_data = src / 'PCBData'

    # ── Validate ───────────────────────────────────────────────────────────────
    if not pcb_data.exists():
        sys.exit(
            f"❌  PCBData/ not found under {src}\n"
            f"    --src should point to the root of the cloned DeepPCB repo.\n"
            f"    Clone with:  git clone https://github.com/tangsanli5201/DeepPCB {src}"
        )

    trainval_txt = pcb_data / 'trainval.txt'
    test_txt     = pcb_data / 'test.txt'
    if not trainval_txt.exists():
        sys.exit(f"❌  PCBData/trainval.txt not found under {src}")

    if dst.exists():
        if args.overwrite:
            print(f"⚠️   Removing existing {dst}")
            shutil.rmtree(dst)
        else:
            sys.exit(
                f"❌  Destination already exists: {dst}\n"
                f"    Add --overwrite to replace it."
            )

    # ── Parse list files ───────────────────────────────────────────────────────
    print(f"\n📂  Source   : {src}")
    print(f"📂  PCBData  : {pcb_data}")
    print(f"📂  Dest     : {dst}")

    trainval_bases = parse_listfile(trainval_txt)
    test_bases     = parse_listfile(test_txt) if test_txt.exists() else []
    print(f"\n📋  trainval.txt : {len(trainval_bases)} entries")
    print(f"📋  test.txt     : {len(test_bases)} entries  "
          f"{'(all → val)' if test_bases else '(not found)'}")

    # ── Train / val split on trainval ──────────────────────────────────────────
    rng = random.Random(args.seed)
    shuffled = list(trainval_bases)
    rng.shuffle(shuffled)

    cut        = int(len(shuffled) * (1.0 - args.val_split))
    train_b    = shuffled[:cut]
    val_from_tv = shuffled[cut:]
    val_b      = val_from_tv + test_bases

    print(f"\n✂️   Split    : {len(train_b)} train pairs  |  "
          f"{len(val_from_tv)} val pairs (from trainval)  +  "
          f"{len(test_bases)} val pairs (from test.txt)  =  "
          f"{len(val_b)} val total")

    # ── Copy loop ──────────────────────────────────────────────────────────────
    stats = {'train': {'defect': 0, 'normal': 0, 'missing': 0},
             'val':   {'defect': 0, 'normal': 0, 'missing': 0}}

    for split, bases in [('train', train_b), ('val', val_b)]:
        prefix = 'tr_' if split == 'train' else 'vl_'
        defect_dir = dst / split / 'defect'
        normal_dir = dst / split / 'normal'
        defect_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)

        for base in bases:
            test_img, temp_img = resolve_pair(pcb_data, base)

            if test_img:
                copy_img(test_img, defect_dir, prefix)
                stats[split]['defect'] += 1
            else:
                stats[split]['missing'] += 1
                print(f"  ⚠️  Missing test image: {base}_test.jpg")

            if temp_img:
                copy_img(temp_img, normal_dir, prefix)
                stats[split]['normal'] += 1
            else:
                stats[split]['missing'] += 1
                print(f"  ⚠️  Missing temp image: {base}_temp.jpg")

        print(f"  [{split}]  defect={stats[split]['defect']}  "
              f"normal={stats[split]['normal']}  "
              f"missing={stats[split]['missing']}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  Dataset prepared successfully")
    print("=" * 55)
    for split in ('train', 'val'):
        nd, nn = stats[split]['defect'], stats[split]['normal']
        total  = nd + nn
        if total:
            print(f"  {split:<6}: {total:5d} images  "
                  f"({100*nd/total:.1f}% defect, {100*nn/total:.1f}% normal)")

    print()
    print("  Ready to use:")
    dst_str = str(dst).replace(os.path.expanduser('~'), '~')
    print(f"    python models/evaluate.py --dataset {dst_str} \\")
    print(f"        --checkpoint models/checkpoints/best_model.pth")
    print(f"    python quantization/quantize_model.py \\")
    print(f"        --checkpoint models/checkpoints/best_model.pth \\")
    print(f"        --dataset {dst_str}")


if __name__ == '__main__':
    main()
