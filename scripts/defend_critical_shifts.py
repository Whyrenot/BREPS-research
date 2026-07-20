"""
defend_critical_shifts.py
=========================
Apply Randomized Smoothing (size perturbation) defence to *bad* bounding boxes
from a critical_shifts JSON file, then compare:

    undefended IoU  (bad_box  → model → mask vs GT)
    defended   IoU  (smooth(bad_box) → model → avg-mask vs GT)
    reference  IoU  (best_box → model → mask vs GT)

Public API (importable):
    CaseResult       – dataclass holding all masks + metrics for one shift case
    ImageResult      – dataclass holding per-image inference results
    run_image(...)   – run full inference for one image, return ImageResult
    evaluate_defence – CLI entry-point that also saves a CSV

Usage (single GPU):
    python heatmaps/defend_critical_shifts.py \\
        --critical_shifts critical_shifts.json \\
        --images_dir /path/to/images \\
        --masks_dir  /path/to/masks  \\
        --checkpoint_path /path/to/sam_vit_b.pth \\
        --model_name SAM --model_type vit_b \\
        --Y 16 --sigma 0.05 --averaging_mode sigmoid \\
        --output_csv outputs_smoothed/defence_results.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
    sample_size_perturbed_boxes,
    sample_size_and_center_perturbed_boxes,
)
from segment_anything.utils.transforms import ResizeLongestSide


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """All data produced for one critical-shift case."""
    image_name: str

    # boxes in SAM-internal 1024-space (as stored in JSON)
    best_box_1024: np.ndarray          # shape (4,)
    bad_box_1024: np.ndarray           # shape (4,)

    # boxes in original image pixel space
    best_box_orig: np.ndarray          # shape (4,)  [x1,y1,x2,y2]
    bad_box_orig: np.ndarray           # shape (4,)

    # Y perturbed boxes used for smoothing, in original image pixel space
    perturbed_boxes_orig: np.ndarray   # shape (Y, 4)

    # predicted binary masks, shape (H, W), dtype bool
    best_mask: np.ndarray              # reference (best_box) mask
    bad_mask: np.ndarray               # undefended (bad_box) mask
    defended_mask: np.ndarray          # randomised-smoothing mask

    # IoUs
    reference_iou: float
    undefended_iou: float
    defended_iou: float

    # from JSON
    best_iou_json: float
    bad_iou_json: float
    iou_drop: float


@dataclass
class ImageResult:
    """All inference results for a single image."""
    image_name: str
    orig_size: tuple                   # (H, W)

    case_results: List[CaseResult]     # one per critical-shift case

    # "final prediction": SAM with the tight GT bounding box
    gt_bbox_orig: np.ndarray           # shape (4,)  [x1,y1,x2,y2]
    final_pred_mask: np.ndarray        # shape (H, W) bool
    final_pred_iou: float


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def boxes_to_original(
    boxes: np.ndarray,
    original_size: tuple,
    target_length: int = 1024,
) -> np.ndarray:
    """SAM-internal 1024-space → original image pixel space."""
    old_h, old_w = original_size
    new_h, new_w = ResizeLongestSide.get_preprocess_shape(old_h, old_w, target_length)
    b = boxes.astype(np.float64, copy=True).reshape(-1, 2, 2)
    b[..., 0] *= float(old_w) / float(new_w)
    b[..., 1] *= float(old_h) / float(new_h)
    return np.round(b.reshape(-1, 4)).astype(np.int32)


def original_to_1024(
    boxes: np.ndarray,
    original_size: tuple,
    target_length: int = 1024,
) -> np.ndarray:
    """Original image pixel space → SAM-internal 1024-space."""
    old_h, old_w = original_size
    new_h, new_w = ResizeLongestSide.get_preprocess_shape(old_h, old_w, target_length)
    b = boxes.astype(np.float64, copy=True).reshape(-1, 2, 2)
    b[..., 0] *= float(new_w) / float(old_w)
    b[..., 1] *= float(new_h) / float(old_h)
    return b.reshape(-1, 4)


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _prepare_image(image_path: str, predictor) -> tuple[int, int]:
    """Load image, resize if needed, encode with predictor. Return (H, W)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[0] > 1024 and img.shape[1] > 1024:
        img = cv2.resize(img, (1024, 1024))
    predictor.set_image(img)
    return img.shape[:2]  # (H, W)


def _predict_single_box(
    box: torch.Tensor,
    predictor,
    rank,
    boxes_already_transformed: bool = False,
) -> torch.Tensor:
    """Run model on a single box; return binary mask (H, W) bool tensor on CPU.

    boxes_already_transformed: if True, skip apply_boxes_torch (box is already
        in SAM's internal coordinate space, as stored in the critical_shifts JSON).
    """
    box_t = box.unsqueeze(0).float()
    if not boxes_already_transformed and hasattr(predictor, "transform"):
        box_t = predictor.transform.apply_boxes_torch(box_t, predictor.original_size)
    box_t = box_t.to(rank)

    with torch.inference_mode():
        try:
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=box_t,
                multimask_output=False,
                return_logits=False,
            )
        except TypeError:
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=box_t,
                multimask_output=False,
            )
    return masks[0, 0].bool().cpu()  # (H, W)


@torch.inference_mode()
def _predict_smoothed_box(
    bad_box: torch.Tensor,
    image_shape: tuple[int, int],
    predictor,
    rank,
    Y: int,
    sigma_w: float,
    sigma_h: float,
    averaging_mode: str,
    sigma_cx: float = 0.0,
    sigma_cy: float = 0.0,
    perturb_mode: str = "size",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:  # (mask, perturbed_boxes)
    """
    Generate Y perturbed copies of bad_box, run through model,
    average masks according to averaging_mode. Returns binary mask (H, W).
    """
    if perturb_mode == "size_center":
        perturbed = sample_size_and_center_perturbed_boxes(
            base_box=bad_box,
            image_shape=image_shape,
            num_samples=Y,
            sigma_w=sigma_w,
            sigma_h=sigma_h,
            sigma_cx=sigma_cx,
            sigma_cy=sigma_cy,
            seed=seed,
        )
    else:
        perturbed = sample_size_perturbed_boxes(
            base_box=bad_box,
            image_shape=image_shape,
            num_samples=Y,
            sigma_w=sigma_w,
            sigma_h=sigma_h,
            seed=seed,
        )  # (Y, 4)

    perturbed_boxes = perturbed.cpu()   # save for return
    perturbed_t = perturbed.float().to(rank)

    try:
        masks_logits, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=perturbed_t,
            multimask_output=False,
            return_logits=True,
        )
    except TypeError:
        masks_logits, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=perturbed_t,
            multimask_output=False,
        )
        return masks_logits[:, 0].float().mean(dim=0).gt(0.5).cpu(), perturbed_boxes

    masks_logits = masks_logits.float()  # (Y, 1, H, W)
    scores = scores.float()              # (Y, 1)

    thresh = 0.0
    if averaging_mode == "score_weighted":
        weights = torch.softmax(scores, dim=0).unsqueeze(-1).unsqueeze(-1)
        avg = (weights * masks_logits).sum(dim=0)
        smoothed = torch.sigmoid(avg) > 0.5
    elif averaging_mode == "best_of_n":
        best_idx = scores[:, 0].argmax().item()
        smoothed = (masks_logits[best_idx] > thresh)
    elif "logit" in averaging_mode:
        avg = masks_logits.mean(dim=0)
        smoothed = torch.sigmoid(avg) > 0.5
    elif "sigmoid" in averaging_mode:
        smoothed = torch.sigmoid(masks_logits).mean(dim=0) > 0.5
    elif "binary" in averaging_mode:
        smoothed = (masks_logits > thresh).float().mean(dim=0) > 0.5
    else:
        avg = masks_logits.mean(dim=0)
        smoothed = torch.sigmoid(avg) > 0.5

    return smoothed[0].cpu(), perturbed_boxes  # (H, W), (Y, 4)


# ---------------------------------------------------------------------------
# File-finding helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MASK_EXTS  = {".png", ".bmp", ".jpg", ".jpeg"}


def _find_file(directory: str, stem: str) -> Optional[Path]:
    """Find any file in *directory* whose stem equals *stem* (case-insensitive)."""
    d = Path(directory)
    for ext in _IMAGE_EXTS | _MASK_EXTS:
        p = d / (stem + ext)
        if p.exists():
            return p
    for p in d.rglob("*"):
        if p.stem.lower() == stem.lower() and p.suffix.lower() in (_IMAGE_EXTS | _MASK_EXTS):
            return p
    return None


# ---------------------------------------------------------------------------
# High-level per-image inference  (PUBLIC API)
# ---------------------------------------------------------------------------

def run_image(
    image_name: str,
    image_path: str,
    mask_path: str,
    shift_cases: list[dict],
    predictor,
    rank,
    Y: int = 16,
    sigma: float = 0.05,
    sigma_center: float = 0.03,
    averaging_mode: str = "sigmoid",
    perturb_mode: str = "size",
) -> ImageResult:
    """
    Run full inference for a single image and return an ImageResult.

    Parameters
    ----------
    image_name   : image stem (e.g. '21077')
    image_path   : full path to source image
    mask_path    : full path to GT mask
    shift_cases  : list of dicts from critical_shifts JSON (already filtered for this image)
    predictor    : loaded SAM predictor
    rank         : torch device
    Y, sigma, ... : smoothing hyper-parameters

    Returns
    -------
    ImageResult with all masks and IoUs filled in.
    """
    # --- load GT mask ---
    gt_mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask_np is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    gt_mask_np = (gt_mask_np > 0)
    if gt_mask_np.shape[0] > 1024 and gt_mask_np.shape[1] > 1024:
        gt_mask_np = cv2.resize(
            gt_mask_np.astype(np.uint8), (1024, 1024), cv2.INTER_NEAREST
        ).astype(bool)

    gt_tensor = torch.from_numpy(gt_mask_np)  # (H, W)

    # --- encode image ---
    orig_size = _prepare_image(str(image_path), predictor)  # (H, W)
    H, W = orig_size

    sam_h, sam_w = getattr(predictor, "input_size", (1024, 1024))
    perturb_shape = (sam_h, sam_w)

    def _iou(pred_mask_bool_tensor):
        return batch_iou_torch(
            gt_tensor.unsqueeze(0).unsqueeze(0),
            pred_mask_bool_tensor.unsqueeze(0).unsqueeze(0),
        ).item()

    # --- per-shift inference ---
    case_results: list[CaseResult] = []
    for case in tqdm(shift_cases, desc=f"  shifts [{image_name}]", leave=False):
        best_box = torch.tensor(case["best_box"], dtype=torch.float32)
        bad_box  = torch.tensor(case["bad_box"],  dtype=torch.float32)

        # predict masks (boxes are already in SAM 1024-space)
        best_mask_t = _predict_single_box(best_box, predictor, rank,
                                          boxes_already_transformed=True)
        bad_mask_t  = _predict_single_box(bad_box,  predictor, rank,
                                          boxes_already_transformed=True)
        defended_mask_t, perturbed_boxes_1024 = _predict_smoothed_box(
            bad_box=bad_box,
            image_shape=perturb_shape,
            predictor=predictor,
            rank=rank,
            Y=Y,
            sigma_w=sigma,
            sigma_h=sigma,
            averaging_mode=averaging_mode,
            sigma_cx=sigma_center,
            sigma_cy=sigma_center,
            perturb_mode=perturb_mode,
            seed=42,
        )

        best_box_orig = boxes_to_original(np.array([case["best_box"]]), orig_size)[0]
        bad_box_orig  = boxes_to_original(np.array([case["bad_box"]]),  orig_size)[0]
        perturbed_boxes_orig = boxes_to_original(
            perturbed_boxes_1024.numpy(), orig_size
        )  # (Y, 4)

        case_results.append(CaseResult(
            image_name=image_name,
            best_box_1024=np.array(case["best_box"]),
            bad_box_1024=np.array(case["bad_box"]),
            best_box_orig=best_box_orig,
            bad_box_orig=bad_box_orig,
            perturbed_boxes_orig=perturbed_boxes_orig,
            best_mask=best_mask_t.numpy(),
            bad_mask=bad_mask_t.numpy(),
            defended_mask=defended_mask_t.numpy(),
            reference_iou=_iou(best_mask_t),
            undefended_iou=_iou(bad_mask_t),
            defended_iou=_iou(defended_mask_t),
            best_iou_json=float(case["best_iou"]),
            bad_iou_json=float(case["bad_iou"]),
            iou_drop=float(case["iou_drop"]),
        ))

    # --- final prediction: GT tight bbox ---
    rmin, rmax, cmin, cmax = get_bbox_from_mask(gt_mask_np.astype(np.uint8))
    gt_bbox_orig = np.array([cmin, rmin, cmax, rmax], dtype=np.int32)
    gt_bbox_1024 = original_to_1024(gt_bbox_orig[None], orig_size)[0]
    gt_box_t = torch.tensor(gt_bbox_1024, dtype=torch.float32)
    final_mask_t = _predict_single_box(gt_box_t, predictor, rank,
                                       boxes_already_transformed=True)

    return ImageResult(
        image_name=image_name,
        orig_size=orig_size,
        case_results=case_results,
        gt_bbox_orig=gt_bbox_orig,
        final_pred_mask=final_mask_t.numpy(),
        final_pred_iou=_iou(final_mask_t),
    )


# ---------------------------------------------------------------------------
# CLI evaluation loop  (unchanged behaviour)
# ---------------------------------------------------------------------------

def evaluate_defence(args) -> list[ImageResult]:
    """Run defence for all cases in the JSON; save CSV; return ImageResult list."""
    with open(args.critical_shifts, "r") as f:
        shifts = json.load(f)
    logger.info(f"Loaded {len(shifts)} critical shift cases from {args.critical_shifts}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rank = device

    predictor = load_model(
        model_name=args.model_name,
        model_type=args.model_type,
        checkpoint=args.checkpoint_path,
        device=device,
    )

    # group cases by image
    from collections import defaultdict
    by_image: dict[str, list[dict]] = defaultdict(list)
    for case in shifts:
        by_image[case["image_name"]].append(case)

    rows: list[dict] = []
    image_results: list[ImageResult] = []

    for image_name, cases in tqdm(by_image.items(), desc="Images"):
        image_path = _find_file(args.images_dir, image_name)
        mask_path  = _find_file(args.masks_dir,  image_name)

        if image_path is None:
            logger.warning(f"Image not found for '{image_name}', skipping.")
            continue
        if mask_path is None:
            logger.warning(f"Mask not found for '{image_name}', skipping.")
            continue

        try:
            img_result = run_image(
                image_name=image_name,
                image_path=str(image_path),
                mask_path=str(mask_path),
                shift_cases=cases,
                predictor=predictor,
                rank=rank,
                Y=args.Y,
                sigma=args.sigma,
                sigma_center=args.sigma_center,
                averaging_mode=args.averaging_mode,
                perturb_mode=args.perturb_mode,
            )
        except Exception as e:
            logger.error(f"Failed on '{image_name}': {e}")
            continue

        image_results.append(img_result)

        for cr in img_result.case_results:
            rows.append({
                "image_name":     cr.image_name,
                "best_iou_json":  cr.best_iou_json,
                "bad_iou_json":   cr.bad_iou_json,
                "iou_drop_json":  cr.iou_drop,
                "reference_iou":  cr.reference_iou,
                "undefended_iou": cr.undefended_iou,
                "defended_iou":   cr.defended_iou,
                "iou_recovery":   cr.defended_iou - cr.undefended_iou,
                "Y":              args.Y,
                "sigma":          args.sigma,
                "averaging_mode": args.averaging_mode,
            })

        logger.debug(
            f"{image_name}: "
            + " | ".join(
                f"undefended={cr.undefended_iou:.3f} defended={cr.defended_iou:.3f}"
                for cr in img_result.case_results
            )
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved {len(df)} rows to {args.output_csv}")

    if not df.empty:
        logger.info(
            f"\n=== Summary ===\n"
            f"  Mean undefended IoU : {df['undefended_iou'].mean():.4f}\n"
            f"  Mean defended   IoU : {df['defended_iou'].mean():.4f}\n"
            f"  Mean IoU recovery   : {df['iou_recovery'].mean():.4f}\n"
            f"  Mean reference  IoU : {df['reference_iou'].mean():.4f}\n"
        )

    return image_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Randomized Smoothing (size-perturbation) defence for critical-shift bad boxes. "
            "Evaluates undefended vs defended IoU for each case in the JSON file."
        )
    )
    parser.add_argument("--critical_shifts", type=str, default="critical_shifts.json")
    parser.add_argument("--images_dir",      type=str, required=True)
    parser.add_argument("--masks_dir",       type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name",      type=str, default="SAM",
                        choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    parser.add_argument("--model_type",      type=str, default="vit_b")
    parser.add_argument("--gpu",             type=int, default=0)
    parser.add_argument("--Y",               type=int, default=16)
    parser.add_argument("--sigma",           type=float, default=0.05)
    parser.add_argument("--sigma_center",    type=float, default=0.03)
    parser.add_argument("--perturb_mode",    type=str, default="size",
                        choices=["size", "size_center"])
    parser.add_argument("--averaging_mode",  type=str, default="sigmoid",
                        choices=["logit", "sigmoid", "binary", "score_weighted", "best_of_n"])
    parser.add_argument("--output_csv",      type=str,
                        default="outputs_smoothed/defence_results.csv")

    args = parser.parse_args()

    from heatmaps.env_dispatch import maybe_dispatch_to_env
    maybe_dispatch_to_env(args.model_name, __file__)

    evaluate_defence(args)
