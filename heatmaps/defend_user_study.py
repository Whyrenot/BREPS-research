"""
defend_user_study.py
====================
Randomized-Smoothing defence applied to *real* user-drawn masks
from the user_study FOR_TEST/ dataset.

For each user_mask we:
    1. Extract user_bbox (tight bbox of the user-drawn mask).
    2. undefended_iou  = IoU(SAM(user_bbox),               GT)
    3. defended_iou    = IoU(smoothed_SAM(user_bbox),      GT)
    4. tight_gt_iou    = IoU(SAM(tight_bbox_from_GT),      GT)   <- upper-bound reference
    5. user_mask_iou   = IoU(user_mask,                    GT)   <- how good was the user's own drawing

Re-uses all smoothing primitives from heatmaps/defend_critical_shifts.py and
heatmaps/comp_hw_smoothed.py — no model logic is duplicated.

Single GPU:
    python heatmaps/defend_user_study.py \\
        --root      /.../datasets/FOR_TEST \\
        --checkpoint_path /.../sam_vit_b_01ec64.pth \\
        --model_name SAM --model_type vit_b \\
        --use mp --Y 16 --sigma 0.05 --averaging_mode sigmoid \\
        --output_csv outputs_smoothed/user_study_defence.csv

Multi-GPU (DDP-style sharding by rank::world_size):
    torchrun --nproc_per_node=8 heatmaps/defend_user_study.py ...
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from heatmaps.comp_hw_smoothed import (
    batch_iou_torch,
    get_bbox_from_mask,
    load_model,
)
from heatmaps.defend_critical_shifts import (
    _predict_single_box,
    _predict_smoothed_box,
    _prepare_image,
    boxes_to_original,
    original_to_1024,
)


# user_mask filename: {stem}_{inst}_{userid}--{attemptid}_{p|m}.png
USER_MASK_RE = re.compile(
    r"^(?P<stem>.+?)_(?P<inst>\d+)_(?P<user>[0-9a-f]+)--(?P<attempt>[0-9a-f]+)_(?P<kind>[pm])\.png$"
)
    

@dataclass
class UserMaskRecord:
    stem: str
    inst: str
    user: str
    attempt: str
    kind: str          # 'p' or 'm'
    path: Path


def index_user_masks(user_masks_dir: Path, kinds: tuple[str, ...]) -> list[UserMaskRecord]:
    records: list[UserMaskRecord] = []
    skipped = 0
    for p in sorted(user_masks_dir.iterdir()):
        if not p.is_file():
            continue
        m = USER_MASK_RE.match(p.name)
        if m is None:
            skipped += 1
            continue
        if m["kind"] not in kinds:
            continue
        records.append(UserMaskRecord(
            stem=m["stem"], inst=m["inst"], user=m["user"],
            attempt=m["attempt"], kind=m["kind"], path=p,
        ))
    if skipped:
        logger.warning(f"{skipped} files in {user_masks_dir} did not match the user_mask filename pattern.")
    return records


def _load_binary_mask(path: Path, target_shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Load a uint8 mask, binarise it (> 0), optionally resize to target_shape (H,W)."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if target_shape is not None and m.shape != target_shape:
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0)


def run(args):
    root = Path(args.root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    user_masks_dir = root / "user_masks"

    # ---- DDP sharding ----
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = args.gpu
        rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"world_size={world_size} rank={rank} device={device}")

    # ---- collect & shard records ----
    kinds = tuple(args.use)
    records = index_user_masks(user_masks_dir, kinds=kinds)
    if rank == 0:
        logger.info(f"Indexed {len(records)} user_masks (kinds={kinds})")

    # group by stem so each image is encoded once per worker
    per_stem: dict[str, list[UserMaskRecord]] = defaultdict(list)
    for r in records:
        per_stem[r.stem].append(r)

    stems = sorted(per_stem.keys())
    stems = stems[rank::world_size]  # shard by image stem

    if args.limit_images > 0:
        stems = stems[: args.limit_images]
    if rank == 0:
        logger.info(f"This worker will process {len(stems)} image stems "
                    f"(total user_masks: {sum(len(per_stem[s]) for s in stems)})")

    # ---- load model once ----
    predictor = load_model(
        model_name=args.model_name,
        model_type=args.model_type,
        checkpoint=args.checkpoint_path,
        device=device,
    )

    rows: list[dict] = []
    for stem in tqdm(stems, desc=f"[r{rank}] images", position=local_rank):
        image_path = images_dir / f"{stem}.png"
        gt_path = masks_dir / f"{stem}.png"
        if not image_path.exists():
            logger.warning(f"missing image: {image_path}")
            continue
        if not gt_path.exists():
            logger.warning(f"missing GT mask: {gt_path}")
            continue

        # encode image (also tells us the working H,W)
        try:
            H, W = _prepare_image(str(image_path), predictor)
        except Exception as e:
            logger.error(f"encode failed for {stem}: {e}")
            continue
        orig_size = (H, W)

        # GT at the same resolution as predictor output
        gt_bin = _load_binary_mask(gt_path, target_shape=orig_size)
        gt_tensor = torch.from_numpy(gt_bin)

        def iou_vs_gt(pred_bool_t: torch.Tensor) -> float:
            return batch_iou_torch(
                gt_tensor.unsqueeze(0).unsqueeze(0),
                pred_bool_t.unsqueeze(0).unsqueeze(0),
            ).item()

        sam_h, sam_w = getattr(predictor, "input_size", (1024, 1024))
        perturb_shape = (sam_h, sam_w)

        # ---- tight GT bbox reference (computed once per image) ----
        if gt_bin.sum() == 0:
            logger.warning(f"empty GT for {stem}, skipping image")
            continue
        rmin, rmax, cmin, cmax = get_bbox_from_mask(gt_bin.astype(np.uint8))
        gt_bbox_orig = np.array([cmin, rmin, cmax, rmax], dtype=np.int32)
        gt_bbox_1024 = original_to_1024(gt_bbox_orig[None], orig_size)[0]
        gt_box_t = torch.tensor(gt_bbox_1024, dtype=torch.float32)
        tight_pred_t = _predict_single_box(gt_box_t, predictor, device,
                                           boxes_already_transformed=True)
        tight_gt_iou = iou_vs_gt(tight_pred_t)

        # ---- per user_mask ----
        for rec in per_stem[stem]:
            try:
                user_mask_bin = _load_binary_mask(rec.path, target_shape=orig_size)
            except Exception as e:
                logger.error(f"failed to read {rec.path}: {e}")
                continue
            if user_mask_bin.sum() == 0:
                continue  # nothing to extract a bbox from

            # bbox from user mask
            rmin, rmax, cmin, cmax = get_bbox_from_mask(user_mask_bin.astype(np.uint8))
            user_bbox_orig = np.array([cmin, rmin, cmax, rmax], dtype=np.int32)
            uw = user_bbox_orig[2] - user_bbox_orig[0]
            uh = user_bbox_orig[3] - user_bbox_orig[1]
            if uw <= 0 or uh <= 0:
                continue

            user_bbox_1024 = original_to_1024(user_bbox_orig[None], orig_size)[0]
            user_box_t = torch.tensor(user_bbox_1024, dtype=torch.float32)

            # undefended
            undef_mask_t, undef_score = _predict_single_box(
                user_box_t, predictor, device,
                boxes_already_transformed=True, return_score=True,
            )
            undef_iou = iou_vs_gt(undef_mask_t)

            # defended (Y perturbed copies, averaged)
            def_mask_t, _perturbed, def_scores, def_best_idx = _predict_smoothed_box(
                bad_box=user_box_t,
                image_shape=perturb_shape,
                predictor=predictor,
                rank=device,
                Y=args.Y,
                sigma_w=args.sigma,
                sigma_h=args.sigma,
                averaging_mode=args.averaging_mode,
                sigma_cx=args.sigma_center,
                sigma_cy=args.sigma_center,
                perturb_mode=args.perturb_mode,
                seed=args.seed,
                include_base_box=args.include_base_box,
                return_scores=True,
                topk=args.topk,
                stability_M=args.stability_M,
                stability_sigma_w=args.stability_sigma,
                stability_sigma_h=args.stability_sigma,
            )
            def_iou = iou_vs_gt(def_mask_t)

            # score stats across all candidates (Y or Y+1 if include_base_box)
            scores_1d = def_scores[:, 0].numpy()
            def_score_max  = float(scores_1d.max())
            def_score_mean = float(scores_1d.mean())
            def_score_min  = float(scores_1d.min())
            # if include_base_box, candidate 0 IS the base box;
            #   base_score_in_pool lets us compare base vs perturbations side-by-side.
            if args.include_base_box:
                base_score_in_pool = float(scores_1d[0])
                base_score_rank    = int((scores_1d >= scores_1d[0]).sum())  # 1=best
                base_won           = bool(def_best_idx == 0)
            else:
                base_score_in_pool = float("nan")
                base_score_rank    = -1
                base_won           = False

            # how close was the user's own drawing?
            user_mask_iou = float(
                np.logical_and(user_mask_bin, gt_bin).sum()
                / max(1, int(np.logical_or(user_mask_bin, gt_bin).sum()))
            )

            rows.append({
                "image_stem":     rec.stem,
                "user":           rec.user,
                "attempt":        rec.attempt,
                "kind":           rec.kind,
                "user_bbox":      user_bbox_orig.tolist(),
                "user_bbox_w":    int(uw),
                "user_bbox_h":    int(uh),
                "user_mask_iou":  user_mask_iou,
                "undefended_iou": undef_iou,
                "defended_iou":   def_iou,
                "iou_recovery":   def_iou - undef_iou,
                "tight_gt_iou":   tight_gt_iou,
                # SAM-predicted scores (model's own confidence ≈ predicted IoU)
                "undef_score":         undef_score,
                "def_score_max":       def_score_max,
                "def_score_mean":      def_score_mean,
                "def_score_min":       def_score_min,
                "base_score_in_pool":  base_score_in_pool,
                "base_score_rank":     base_score_rank,
                "base_won":            base_won,
                "def_best_idx":        def_best_idx,
                "Y":              args.Y,
                "sigma":          args.sigma,
                "sigma_center":   args.sigma_center,
                "perturb_mode":   args.perturb_mode,
                "averaging_mode": args.averaging_mode,
                "include_base_box": args.include_base_box,
            })

    # ---- save per-worker CSV ----
    out_path = Path(args.output_csv)
    if world_size > 1:
        out_path = out_path.with_name(f"{out_path.stem}.rank{rank}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[r{rank}] wrote {len(df)} rows → {out_path}")

    if not df.empty and rank == 0:
        score_block = ""
        if args.include_base_box and "base_score_in_pool" in df.columns:
            n_pool = (
                df["Y"].iloc[0] + 1
                if args.include_base_box else df["Y"].iloc[0]
            )
            score_block = (
                f"\n  --- SAM score diagnostics (with base in pool of {n_pool}) ---\n"
                f"  base_score        mean : {df['base_score_in_pool'].mean():.4f}\n"
                f"  perturb_score_max mean : {df['def_score_max'].mean():.4f}\n"
                f"  base_score_rank   mean : {df['base_score_rank'].mean():.2f}   "
                f"(1 = base wins, {n_pool} = base last)\n"
                f"  base wins (best_of_n)  : {df['base_won'].mean():.2%}\n"
            )
        else:
            score_block = (
                f"\n  --- SAM score diagnostics ---\n"
                f"  undef_score       mean : {df['undef_score'].mean():.4f}\n"
                f"  def_score_max     mean : {df['def_score_max'].mean():.4f}\n"
                f"  perturbation has higher score: "
                f"{(df['def_score_max'] > df['undef_score']).mean():.2%}\n"
            )
        logger.info(
            "\n=== Summary (this worker) ===\n"
            f"  user_masks evaluated  : {len(df)}\n"
            f"  user_mask_iou  mean   : {df['user_mask_iou'].mean():.4f}\n"
            f"  undefended_iou mean   : {df['undefended_iou'].mean():.4f}\n"
            f"  defended_iou   mean   : {df['defended_iou'].mean():.4f}\n"
            f"  iou_recovery   mean   : {df['iou_recovery'].mean():+.4f}\n"
            f"  tight_gt_iou   mean   : {df['tight_gt_iou'].mean():.4f}\n"
            f"  hurts ratio (rec<0)   : {(df['iou_recovery'] < 0).mean():.2%}\n"
            f"  big-help (rec>0.1)    : {(df['iou_recovery'] > 0.1).mean():.2%}"
            + score_block
        )

    return df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True,
                   help="FOR_TEST/ root (contains images/, masks/, user_masks/)")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_name", default="SAM",
                   choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU id when running single-process (ignored under torchrun)")

    p.add_argument("--use", default="mp",
                   help="Which user_mask kinds to evaluate: 'm', 'p', or 'mp' (default: both)")
    p.add_argument("--limit_images", type=int, default=0,
                   help="Process only first N image stems (after sharding). 0 = all.")

    # smoothing hyper-parameters
    p.add_argument("--Y", type=int, default=16,
                   help="Number of perturbed boxes per defence call")
    p.add_argument("--sigma", type=float, default=0.05,
                   help="Std-dev of relative width/height perturbation")
    p.add_argument("--sigma_center", type=float, default=0.03,
                   help="Std-dev of relative center perturbation (used iff perturb_mode=size_center)")
    p.add_argument("--perturb_mode", default="size",
                   choices=["size", "size_center"])
    p.add_argument("--averaging_mode", default="sigmoid",
                   choices=["logit", "sigmoid", "binary", "score_weighted",
                            "best_of_n", "saif_score", "saif_stability"])
    p.add_argument("--topk", type=int, default=3,
                   help="SAIF modes: keep top-k most confident candidate boxes")
    p.add_argument("--stability_M", type=int, default=6,
                   help="saif_stability: micro-perturbations per candidate for "
                        "the variability/stability estimate")
    p.add_argument("--stability_sigma", type=float, default=None,
                   help="saif_stability: sigma of the micro-perturbations "
                        "(default = --sigma)")
    p.add_argument("--include_base_box", action="store_true", default=False,
                   help="Prepend the unperturbed bad_box to the candidate pool "
                        "(useful with --averaging_mode best_of_n: worst case becomes "
                        "the undefended prediction).")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_csv", default="outputs_smoothed/user_study_defence.csv")
    args = p.parse_args()

    # normalize --use
    args.use = "".join(sorted(set(args.use.lower())))
    for ch in args.use:
        if ch not in ("p", "m"):
            raise SystemExit(f"--use must be subset of 'pm', got {args.use!r}")

    return args


if __name__ == "__main__":
    from heatmaps.env_dispatch import maybe_dispatch_to_env

    _args = parse_args()
    maybe_dispatch_to_env(_args.model_name, __file__)
    run(_args)
