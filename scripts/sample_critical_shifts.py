"""
sample_critical_shifts.py
=========================
Critical shifts on a representative sample of the user study, for one model.

For each FOR_TEST subset (the 10 of steer_user_study.DATASETS, medical
included) the images are put in a fixed random order (--seed). Each image gets
the full w x h IoU heatmap (heatmaps/comp_hw.py under torchrun) and up to
--max_cases critical shifts (find_critical_shifts.find_critical_shift, its
default thresholds). An image without a single critical shift is passed over
and the next one in the order is taken, until --n_images images of the subset
have at least one -- or the subset runs out, which summary.csv reports.

The order depends only on --seed and the subset's file names, so every model
walks the same sequence; the selected images can still differ between models
(one model breaks where another does not).

Images go to torchrun in rounds of ceil(need / nproc) * nproc, so no GPU idles
on a tail of one or two images; the selection is still the first --n_images
successes in the order, as if images were tried one at a time.

Re-running the same command resumes: CSVs already in --out_dir are reused
(comp_hw --continue_from), a CSV cut off by a killed run is recomputed.

SAM-3 re-execs itself into the sam3 conda env (heatmaps/env_dispatch.py), so
torchrun runs from that env too.

--check (on unless --no_check) runs comp_hw's batched path on the first image
in this process and compares its masks with the predictor's own public
predict() on the same pixel boxes, and the CSV box frame with SAM-1's: a wrong
box frame fails here in a minute instead of after a night of heatmaps. It also
times a few batches and prints the expected time per image.

Output (--out_dir, one model):
  <dataset>/res_final_<stem>_.csv   every box of the grid with its IoU (SAM-1 1024 frame)
  critical_shifts_<dataset>.json    cases of the selected images: find_critical_shifts
                                    format plus "dataset", biggest drop first
  critical_shifts_all.json          all subsets in one file; images_dir / masks_dir for
                                    it are FOR_TEST/images and FOR_TEST/masks
  selection.json                    per subset: the order, cases per tried image, selection
  summary.csv                       per subset: tried, without cases, selected, cases

Example:
    python scripts/sample_critical_shifts.py --model_name SAM \\
        --root /.../user_study/FOR_TEST \\
        --checkpoint_path /.../MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth \\
        --out_dir exp_res/critical_shifts_sample/SAM
    # all three models in a row: scripts/run_sample_critical_shifts.sh
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from heatmaps import comp_hw as CH  # noqa: E402
from heatmaps.env_dispatch import maybe_dispatch_to_env  # noqa: E402
from find_critical_shifts import find_critical_shift  # noqa: E402  (scripts/ is sys.path[0])

# the user-study subsets, as scripts/steer_user_study.DATASETS
DATASETS = OrderedDict([("COCO", "COCO_MVal_"), ("Berkeley", "Berkeley_"), ("GrabCut", "GrabCut_"),
                        ("DAVIS", "DAVIS_"), ("PASCAL", "PASCAL_"), ("ADE20K", "ADE20K_"), ("TETRIS", "TETRIS_"),
                        ("ACDC", "ACDC_"), ("BUID", "BUID_"), ("MedScribble", "MedScribble_")])
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# a round that fails (on a shared GPU: usually another job's memory) is rerun
ROUND_ATTEMPTS = 3
RETRY_WAIT_S = 120


def index_dir(d: Path) -> dict[str, Path]:
    """stem -> file, first by name when one stem has several extensions."""
    out: dict[str, Path] = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.setdefault(p.stem, p)
    return out


def image_order(stems: list[str], seed: int) -> list[str]:
    return [str(s) for s in np.random.default_rng(seed).permutation(sorted(stems))]


def plan(order: list[str], n_cases: dict[str, int], n_images: int, nproc: int, max_tried: int = 0):
    """(selected, next_round). selected: the images with >= 1 case, in order,
    among the tried prefix of the order; next_round: the untried images to
    compute next, [] once n_images are selected, nothing is left or
    max_tried (> 0) images have been tried."""
    selected = []
    for s in order:
        if s not in n_cases:
            break
        if n_cases[s] > 0:
            selected.append(s)
            if len(selected) == n_images:
                return selected, []
    untried = [s for s in order if s not in n_cases]
    if max_tried > 0:
        untried = untried[:max(0, max_tried - (len(order) - len(untried)))]
    if not untried:
        return selected, []
    need = n_images - len(selected)
    return selected, untried[:math.ceil(need / nproc) * nproc]


def readable(csv: Path) -> bool:
    try:
        return len(pd.read_csv(csv, usecols=["bbox", "iou"])) > 0
    except Exception:  # noqa: BLE001 -- truncated / empty file
        return False


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum().item()
    return (a & b).sum().item() / union if union else 1.0


def run_round(args, stems, imgs, masks, out: Path, nproc: int) -> None:
    """comp_hw on these images only: symlinks in a scratch dir, masks named
    <stem>.png whatever their format (cv2 reads by content)."""
    rd = out / "_round"
    shutil.rmtree(rd, ignore_errors=True)
    for sub in ("images", "masks"):
        (rd / sub).mkdir(parents=True)
    for s in stems:
        (rd / "images" / imgs[s].name).symlink_to(imgs[s].resolve())
        (rd / "masks" / f"{s}.png").symlink_to(masks[s].resolve())
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           str(_REPO_ROOT / "heatmaps" / "comp_hw.py"),
           "--images_dir", str(rd / "images"), "--masks_dir", str(rd / "masks"), "--mask_ext", ".png",
           "--output_dir", str(out), "--model_name", args.model_name, "--model_type", args.model_type,
           "--checkpoint_path", args.checkpoint_path, "--batch_size", str(args.batch_size),
           "--continue_from"]
    for attempt in range(1, ROUND_ATTEMPTS + 1):
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            break
        if attempt == ROUND_ATTEMPTS:
            raise RuntimeError(f"comp_hw failed {ROUND_ATTEMPTS} times on this round (exit {rc})")
        # images finished before the failure keep their CSVs (--continue_from)
        print(f"[warn] comp_hw failed (exit {rc}), attempt {attempt}/{ROUND_ATTEMPTS}; "
              f"retrying in {RETRY_WAIT_S} s", flush=True)
        time.sleep(RETRY_WAIT_S)
    shutil.rmtree(rd)


def _comp_hw_masks(img: Path, predictor, boxes: np.ndarray, device: torch.device, batch_size: int):
    """(masks (N, H, W) bool, CSV boxes (N, 4)) from comp_hw's own path."""
    out = list(CH.multi_bbox_inference(str(img), predictor, boxes, rank=device, batch_size=batch_size))
    m = torch.cat([r["masks"] for r in out]).bool().cpu()
    return m.reshape(len(boxes), *m.shape[-2:]), torch.cat([r["boxes"].cpu().float() for r in out])


def check_model(args, img: Path, mask: Path, device: torch.device) -> None:
    """comp_hw's path against the predictor's own predict() on the same pixel
    boxes (the tight GT box + 7 grid boxes), one box per call and as one
    batch, then a timing. One box per call must match: that is the box
    frame. A batch may differ a little where the model is unsure of a box
    (batch-dependent TF32 / cuDNN kernels), not more."""
    predictor = CH.load_model(model_name=args.model_name, model_type=args.model_type,
                              checkpoint=args.checkpoint_path, device=device)
    gt = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if gt.shape[0] > 1024 and gt.shape[1] > 1024:  # as comp_hw.evaluate_mask
        gt = cv2.resize(gt, (1024, 1024), cv2.INTER_LINEAR)
    gt = gt > 0
    rmin, rmax, cmin, cmax = CH.get_bbox_from_mask(gt)
    H, W = gt.shape
    grid = CH.generate_bounding_boxes(x=(cmax - cmin) // 2 + cmin, y=(rmax - rmin) // 2 + rmin,
                                      height=H, width=W, max_h=2 * H, max_w=2 * W)
    rng = np.random.default_rng(0)
    boxes = np.concatenate([np.array([[cmin, rmin, cmax, rmax]], dtype=np.int32),
                            grid[rng.choice(len(grid), 7, replace=False)]])

    m_batch, csv_boxes = _comp_hw_masks(img, predictor, boxes, device, len(boxes))
    m_one, _ = _comp_hw_masks(img, predictor, boxes, device, 1)
    frame_err = (csv_boxes - CH.boxes_to_1024(torch.as_tensor(boxes), tuple(m_batch.shape[-2:]))).abs().max().item()
    m_pub = []
    with torch.inference_mode():
        for b in boxes:
            m, _, _ = predictor.predict(point_coords=None, point_labels=None,
                                        box=b.astype(np.float32), multimask_output=False)
            m_pub.append(torch.from_numpy(np.asarray(m[0]) > 0))
    agree_one = [_iou(a, b) for a, b in zip(m_one, m_pub)]
    agree = [_iou(a, b) for a, b in zip(m_batch, m_pub)]
    gt_t = torch.from_numpy(gt)
    tight_iou = _iou(m_batch[0], gt_t) if m_batch.shape[-2:] == gt_t.shape else float("nan")
    print(f"[check] {args.model_name} on {img.name} (TF32 matmul "
          f"{'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}): mask IoU vs predict(), one box per call "
          f"min {min(agree_one):.4f}; batch of {len(boxes)} min {min(agree):.4f} "
          f"(all: {', '.join(f'{a:.3f}' for a in agree)}); CSV frame vs SAM-1 1024 frame max err "
          f"{frame_err:.2e}; tight GT box IoU {tight_iou:.3f}")
    if min(agree_one) < 0.995 or frame_err > 1e-3 or min(agree) < 0.95:
        raise SystemExit(f"[check] {args.model_name}: comp_hw's path disagrees with the model's own predict() "
                         f"-- not running the heatmaps")
    if min(agree) < 0.98:
        print("[check] warning: one box per call matches predict(), a batch slightly less: batch-dependent "
              "numerics, not the box frame")
    if not tight_iou > 0.5:
        print(f"[check] warning: the tight GT box gives IoU {tight_iou:.3f} on {img.name}")

    # one untimed batch first: the first big batch picks kernels and grows the allocator
    n_t = min(len(grid), 4 * args.batch_size)
    warm = grid[rng.choice(len(grid), min(len(grid), args.batch_size), replace=False)]
    for _ in CH.multi_bbox_inference(str(img), predictor, warm, rank=device, batch_size=args.batch_size):
        pass
    sub = grid[rng.choice(len(grid), n_t, replace=False)]
    gt_dev = gt_t.to(device)[None, None]
    torch.cuda.synchronize(device)
    t0 = time.time()
    for r in CH.multi_bbox_inference(str(img), predictor, sub, rank=device, batch_size=args.batch_size):
        CH.batch_iou_torch(r["masks"], gt_dev.repeat_interleave(r["masks"].size(0), dim=0)).cpu()
    torch.cuda.synchronize(device)
    per_box = (time.time() - t0) / n_t
    print(f"[check] {n_t} boxes in {per_box * n_t:.1f} s at batch {args.batch_size}: this image's grid "
          f"({len(grid)} boxes) ~{len(grid) * per_box / 60:.1f} min on one GPU")
    del predictor
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    maybe_dispatch_to_env(args.model_name, __file__)

    root, out_dir = Path(args.root), Path(args.out_dir)
    imgs, masks = index_dir(root / "images"), index_dir(root / "masks")
    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in names if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown datasets {unknown}; known: {list(DATASETS)}")
    nproc = args.nproc or torch.cuda.device_count()
    if nproc < 1:
        raise SystemExit("no GPU visible")
    out_dir.mkdir(parents=True, exist_ok=True)
    params = dict(dist_thresh=args.dist_thresh, iou_drop_thresh=args.iou_drop_thresh, max_cases=args.max_cases)

    orders, no_mask = {}, {}
    for name in names:
        stems = [s for s in imgs if s.startswith(DATASETS[name])]
        no_mask[name] = [s for s in stems if s not in masks]
        orders[name] = image_order([s for s in stems if s in masks], args.seed)
        print(f"{name}: {len(orders[name])} images" + (f", {len(no_mask[name])} without a GT mask"
                                                         if no_mask[name] else ""))
        if not orders[name]:
            print(f"[warn] {name}: no images with prefix {DATASETS[name]!r} under {root / 'images'}")

    if not args.no_check:
        first = next(o[0] for o in orders.values() if o)
        torch.cuda.set_device(0)
        check_model(args, imgs[first], masks[first], torch.device("cuda:0"))
        if args.check_only:
            return

    selection = {"model": args.model_name, "checkpoint": args.checkpoint_path, "seed": args.seed,
                 "n_images": args.n_images, "find_params": params, "datasets": {}}
    all_cases, rows = [], []
    for name in names:
        order, out = orders[name], out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        n_cases, cases, skipped = {}, {}, []

        def csv_of(s):
            return Path(CH.get_output_path(str(out), imgs[s].name, random_boxes=False))

        def evaluate(s):
            csv = csv_of(s)
            if not csv.exists():  # comp_hw skipped it (empty / unreadable mask)
                n_cases[s], cases[s] = 0, []
                skipped.append(s)
                return
            if not readable(csv):
                raise RuntimeError(f"{csv}: unreadable right after comp_hw")
            cases[s] = find_critical_shift(csv, **params)
            n_cases[s] = len(cases[s])

        for s in order:  # resume
            if csv_of(s).exists():
                if readable(csv_of(s)):
                    evaluate(s)
                else:
                    print(f"[resume] {csv_of(s)} is cut off, recomputing")
                    csv_of(s).unlink()
        if n_cases:
            print(f"[resume] {name}: {len(n_cases)} images already computed")

        rnd = 0
        while True:
            selected, nxt = plan(order, n_cases, args.n_images, nproc, args.max_tried)
            if not nxt:
                break
            rnd += 1
            print(f"=== {args.model_name} / {name}: round {rnd}, {len(nxt)} images "
                  f"(selected so far {len(selected)}/{args.n_images}) {time.strftime('%H:%M:%S')}", flush=True)
            t0 = time.time()
            run_round(args, nxt, imgs, masks, out, nproc)
            for s in nxt:
                evaluate(s)
            print(f"=== round {rnd} took {(time.time() - t0) / 60:.1f} min; cases per image: "
                  + ", ".join(f"{s}={n_cases[s]}" for s in nxt), flush=True)

        sel_cases = [dict(c, dataset=name) for s in selected for c in cases[s]]
        sel_cases.sort(key=lambda c: c["iou_drop"], reverse=True)
        (out_dir / f"critical_shifts_{name}.json").write_text(json.dumps(sel_cases, indent=4))
        all_cases += sel_cases
        tried = [s for s in order if s in n_cases]
        selection["datasets"][name] = {"order": order, "cases_per_tried_image": {s: n_cases[s] for s in tried},
                                       "skipped_by_comp_hw": skipped, "no_gt_mask": no_mask[name],
                                       "selected": selected}
        (out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
        rows.append({"dataset": name, "images": len(order), "tried": len(tried),
                     "without_cases": sum(n_cases[s] == 0 for s in tried), "skipped_by_comp_hw": len(skipped),
                     "selected": len(selected), "cases": len(sel_cases), "short": len(selected) < args.n_images})
        if len(selected) < args.n_images:
            print(f"[warn] {name}: {len(selected)} images with a critical shift after trying "
                  f"{len(tried)} of {len(order)}")

    all_cases.sort(key=lambda c: c["iou_drop"], reverse=True)
    (out_dir / "critical_shifts_all.json").write_text(json.dumps(all_cases, indent=4))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    p.add_argument("--root", required=True, help="user study FOR_TEST (images/, masks/)")
    p.add_argument("--checkpoint_path", default="",
                   help="SAM-3: '' = the builder takes the weights from the HF cache")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated subsets")
    p.add_argument("--n_images", type=int, default=20, help="images with >= 1 critical shift per subset")
    p.add_argument("--seed", type=int, default=0, help="seed of the image order (same for every model)")
    p.add_argument("--max_tried", type=int, default=0,
                   help="stop a subset after this many tried images, found or not (smoke: 1); 0 = no cap")
    p.add_argument("--dist_thresh", type=float, default=10.0, help="find_critical_shifts, 1024 frame")
    p.add_argument("--iou_drop_thresh", type=float, default=0.75, help="find_critical_shifts")
    p.add_argument("--max_cases", type=int, default=5, help="find_critical_shifts, per image")
    p.add_argument("--batch_size", type=int, default=512, help="comp_hw boxes per decoder call")
    p.add_argument("--nproc", type=int, default=0, help="torchrun processes; 0 = every visible GPU")
    p.add_argument("--no_check", action="store_true", help="skip the batched-path check")
    p.add_argument("--check_only", action="store_true", help="only the check and the timing")
    return p.parse_args()


if __name__ == "__main__":
    main()
