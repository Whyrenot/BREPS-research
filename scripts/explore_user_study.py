"""
Tiny exploration script for the user_study FOR_TEST/ dataset.

Prints just enough info to design defend_user_study.py:
  1. How many user_masks per (image_stem, instance_idx) pair
  2. What `_p` vs `_m` suffix corresponds to (mask area / sparsity)
  3. Whether GT mask is multi-instance (multiple non-zero pixel values)
     or binary (single instance per file)
  4. Pixel-value overlap between user_mask and GT for one example

Run on the server:
    python scripts/explore_user_study.py \\
        --root /home/jovyan/shares/SR006.nfs2/pishugin/rclicks/datasets/FOR_TEST
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


# user_mask filename: {stem}_{inst}_{userid}--{attemptid}_{p|m}.png
USER_MASK_RE = re.compile(
    r"^(?P<stem>.+?)_(?P<inst>\d+)_(?P<user>[0-9a-f]+)--(?P<attempt>[0-9a-f]+)_(?P<kind>[pm])\.png$"
)


def parse_user_mask_name(name: str):
    m = USER_MASK_RE.match(name)
    if not m:
        return None
    return m.groupdict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Path to FOR_TEST/ (contains images/, masks/, user_masks/)")
    parser.add_argument("--example_stem", type=str, default=None,
                        help="If given, show detailed info for this image stem (e.g. ACDC_10101)")
    args = parser.parse_args()

    root = Path(args.root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    user_masks_dir = root / "user_masks"

    image_names = sorted(p.name for p in images_dir.iterdir() if p.is_file())
    mask_names = sorted(p.name for p in masks_dir.iterdir() if p.is_file())
    user_mask_names = sorted(p.name for p in user_masks_dir.iterdir() if p.is_file())

    print("=" * 70)
    print(f"images:     {len(image_names)}  (e.g. {image_names[:3]})")
    print(f"masks:      {len(mask_names)}   (e.g. {mask_names[:3]})")
    print(f"user_masks: {len(user_mask_names)}")
    print("=" * 70)

    # --- 1. parse user_mask filenames ---
    parsed = []
    bad = 0
    for n in user_mask_names:
        p = parse_user_mask_name(n)
        if p is None:
            bad += 1
            continue
        parsed.append(p)

    print(f"\n[1] Filename parsing: {len(parsed)} ok, {bad} unparsed "
          f"(regex: {{stem}}_{{inst}}_{{user}}--{{attempt}}_{{p|m}}.png)")

    # how many per (stem, inst) ?
    per_pair = Counter((p["stem"], p["inst"]) for p in parsed)
    n_pairs = len(per_pair)
    counts = list(per_pair.values())
    print(f"    unique (stem,inst) pairs: {n_pairs}")
    print(f"    user_masks per pair: min={min(counts)} "
          f"median={int(np.median(counts))} max={max(counts)} "
          f"mean={np.mean(counts):.1f}")

    # how many distinct instance indices per image stem?
    per_stem_insts = defaultdict(set)
    for p in parsed:
        per_stem_insts[p["stem"]].add(p["inst"])
    inst_counts = [len(v) for v in per_stem_insts.values()]
    print(f"    distinct instances per image stem: "
          f"min={min(inst_counts)} max={max(inst_counts)} "
          f"mean={np.mean(inst_counts):.2f}")
    multi_inst_stems = [s for s, ii in per_stem_insts.items() if len(ii) > 1]
    print(f"    images with >1 instance: {len(multi_inst_stems)} "
          f"(e.g. {multi_inst_stems[:3]})")

    # how many p vs m ?
    kind_counts = Counter(p["kind"] for p in parsed)
    print(f"    suffix counts: {dict(kind_counts)}")

    # --- 2. GT mask format: how many unique non-zero pixel values? ---
    print(f"\n[2] GT mask pixel values (sampling 10 random masks)")
    rng = np.random.default_rng(0)
    sample = rng.choice(mask_names, size=min(10, len(mask_names)), replace=False)
    for name in sample:
        m = cv2.imread(str(masks_dir / name), cv2.IMREAD_UNCHANGED)
        if m is None:
            print(f"    {name}: <unreadable>")
            continue
        uniq = np.unique(m)
        print(f"    {name}: shape={m.shape} dtype={m.dtype} unique={uniq[:10].tolist()}"
              + ("..." if len(uniq) > 10 else ""))

    # --- 3. Example: take one (stem, inst), inspect _p and _m for same attempt ---
    if args.example_stem is None and parsed:
        # pick stem with both _p and _m
        by_stem = defaultdict(lambda: {"p": [], "m": []})
        for p in parsed:
            by_stem[p["stem"]][p["kind"]].append(p)
        chosen = next(
            (s for s, d in by_stem.items() if d["p"] and d["m"]),
            parsed[0]["stem"],
        )
        args.example_stem = chosen

    print(f"\n[3] Example deep-dive: stem='{args.example_stem}'")
    examples = [p for p in parsed if p["stem"] == args.example_stem]
    if not examples:
        print(f"    no user_masks for stem='{args.example_stem}'")
    else:
        ex_p = next((p for p in examples if p["kind"] == "p"), None)
        ex_m = next((p for p in examples if p["kind"] == "m"), None)

        # GT
        gt_path = masks_dir / f"{args.example_stem}.png"
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            gt_uniq = np.unique(gt)
            print(f"    GT  {gt_path.name}: shape={gt.shape} unique={gt_uniq.tolist()}")

        for tag, ex in [("_p", ex_p), ("_m", ex_m)]:
            if ex is None:
                print(f"    no {tag} example for this stem")
                continue
            fname = (f"{ex['stem']}_{ex['inst']}_{ex['user']}--{ex['attempt']}_{ex['kind']}.png")
            path = user_masks_dir / fname
            um = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if um is None:
                print(f"    {tag} {fname}: <unreadable>")
                continue
            uniq = np.unique(um)
            nz = int((um > 0).sum())
            total = int(um.size)
            ys, xs = np.where(um > 0)
            if len(xs):
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                bbox, bw, bh = (0, 0, 0, 0), 0, 0
            print(f"    {tag} {fname}")
            print(f"       shape={um.shape} dtype={um.dtype} unique={uniq[:5].tolist()}")
            print(f"       nonzero={nz}/{total} ({100*nz/total:.2f}%)  "
                  f"bbox={bbox}  bbox_wh=({bw},{bh})")

            # IoU vs GT (treat GT as binary for now; we'll refine if multi-instance)
            if gt_path.exists():
                gt_bin = (gt > 0)
                um_bin = (um > 0)
                if gt_bin.shape == um_bin.shape:
                    inter = int((gt_bin & um_bin).sum())
                    union = int((gt_bin | um_bin).sum())
                    iou = inter / union if union else 0.0
                    print(f"       IoU(user_mask, GT>0) = {iou:.4f}")
                else:
                    print(f"       SHAPE MISMATCH: GT {gt_bin.shape} vs user {um_bin.shape}")

    print("\nDone. Send the full stdout back.")


if __name__ == "__main__":
    main()
