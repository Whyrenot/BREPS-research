"""
visualize_defence_cases.py
==========================
Visualise the Randomized-Smoothing defence on the WORST and the BEST user box
of a single image from the user_study FOR_TEST/ dataset.

For the chosen image we:
    1. compute undefended_iou for every user box (= IoU(SAM(user_bbox), GT)),
    2. pick the worst box (min undefended_iou) and the best box (max),
    3. for each of the two, render a 2-panel row:
         - left  : UNDEFENDED  -> SAM mask + the user's bbox      + IoU label
         - right : DEFENDED    -> smoothed mask + the best picked  + IoU label
                                  perturbed bbox (best_of_n's choice)
    => a 2x2 figure  (rows = worst / best ; cols = undefended / defended)

All model logic is re-used from defend_user_study / defend_critical_shifts;
nothing is duplicated.

Example (matches your phase-1 run, best_of_n):
    CUDA_VISIBLE_DEVICES=3 python scripts/visualize_defence_cases.py \
        --root ../user_study/FOR_TEST \
        --checkpoint_path /.../sam_vit_b_01ec64.pth \
        --model_name SAM --model_type vit_b \
        --use mp --Y 16 --sigma 0.05 --sigma_center 0.03 \
        --averaging_mode best_of_n \
        --out visualizations/defence_cases.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # headless / server
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from heatmaps.comp_hw_smoothed import batch_iou_torch, get_bbox_from_mask, load_model
from heatmaps.defend_critical_shifts import (
    _predict_single_box,
    _predict_smoothed_box,
    _prepare_image,
    boxes_to_original,
    original_to_1024,
)
from heatmaps.defend_user_study import _load_binary_mask, index_user_masks


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def _load_display_image(image_path: Path) -> np.ndarray:
    """Reload the RGB image at the SAME working resolution _prepare_image uses
    (resize to 1024x1024 only if both dims > 1024), so masks/boxes line up."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[0] > 1024 and img.shape[1] > 1024:
        img = cv2.resize(img, (1024, 1024))
    return img


def _overlay_mask(img: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend a boolean mask onto an RGB uint8 image."""
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = (1.0 - alpha) * out[..., c][m] + alpha * float(color[c])
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_gt_contour(ax, gt_bin: np.ndarray, color="yellow", lw=1.5):
    cnts, _ = cv2.findContours(
        gt_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in cnts:
        c = c.reshape(-1, 2)
        if len(c) >= 2:
            ax.plot(
                np.append(c[:, 0], c[0, 0]),
                np.append(c[:, 1], c[0, 1]),
                color=color, lw=lw, label="_gt",
            )


def _draw_bbox(ax, bbox_xyxy, color, label, ls="-"):
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor=color, lw=2.2, ls=ls,
    ))
    ax.text(x0, max(0, y0 - 4), label, color="black", fontsize=8,
            va="bottom", ha="left",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1.0))


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def collect_user_boxes(stem, per_stem, images_dir, masks_dir, predictor, device):
    """Encode the image, compute undefended IoU for every user box of `stem`.

    Returns (disp_img, gt_bin, orig_size, records[]) where each record holds
    the user bbox (orig + 1024 space), the undefended mask and IoU.
    """
    image_path = images_dir / f"{stem}.png"
    gt_path = masks_dir / f"{stem}.png"
    if not image_path.exists():
        raise SystemExit(f"missing image: {image_path}")
    if not gt_path.exists():
        raise SystemExit(f"missing GT mask: {gt_path}")

    H, W = _prepare_image(str(image_path), predictor)
    orig_size = (H, W)
    gt_bin = _load_binary_mask(gt_path, target_shape=orig_size)
    gt_tensor = torch.from_numpy(gt_bin)
    disp_img = _load_display_image(image_path)

    def iou_vs_gt(pred_bool_t: torch.Tensor) -> float:
        return batch_iou_torch(
            gt_tensor.unsqueeze(0).unsqueeze(0),
            pred_bool_t.unsqueeze(0).unsqueeze(0),
        ).item()

    out = []
    for rec in per_stem[stem]:
        try:
            user_mask_bin = _load_binary_mask(rec.path, target_shape=orig_size)
        except Exception:
            continue
        if user_mask_bin.sum() == 0:
            continue
        rmin, rmax, cmin, cmax = get_bbox_from_mask(user_mask_bin.astype(np.uint8))
        user_bbox_orig = np.array([cmin, rmin, cmax, rmax], dtype=np.int32)
        if (user_bbox_orig[2] - user_bbox_orig[0]) <= 0 or (user_bbox_orig[3] - user_bbox_orig[1]) <= 0:
            continue
        user_bbox_1024 = original_to_1024(user_bbox_orig[None], orig_size)[0]
        user_box_t = torch.tensor(user_bbox_1024, dtype=torch.float32)

        undef_mask_t = _predict_single_box(
            user_box_t, predictor, device, boxes_already_transformed=True
        )
        out.append({
            "rec": rec,
            "user_bbox_orig": user_bbox_orig,
            "user_box_t": user_box_t,
            "undef_mask": undef_mask_t.numpy(),
            "undef_iou": iou_vs_gt(undef_mask_t),
            "iou_fn": iou_vs_gt,
        })
    if not out:
        raise SystemExit(f"no usable user boxes for image {stem}")
    return disp_img, gt_bin, orig_size, out


def defend(box_info, predictor, device, orig_size, args):
    """Run the smoothing defence on one user box; return defended mask, IoU and
    the best picked perturbed bbox (in original-image coords, or None)."""
    sam_h, sam_w = getattr(predictor, "input_size", (1024, 1024))
    def_mask_t, perturbed, scores, best_idx = _predict_smoothed_box(
        bad_box=box_info["user_box_t"],
        image_shape=(sam_h, sam_w),
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
    )
    def_iou = box_info["iou_fn"](def_mask_t)

    best_bbox_orig = None
    if best_idx >= 0:  # best_of_n picked a single concrete box
        best_bbox_orig = boxes_to_original(
            perturbed[best_idx].unsqueeze(0).numpy(), orig_size
        )[0]
    return def_mask_t.numpy(), def_iou, best_bbox_orig, best_idx


def render(disp_img, gt_bin, worst, best, args, stem, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    UNDEF_COLOR = (220, 50, 50)    # red mask
    DEF_COLOR = (40, 200, 90)      # green mask
    USER_BOX = "deepskyblue"
    BEST_BOX = "lime"

    for row, (label, case) in enumerate([("WORST", worst), ("BEST", best)]):
        rec = case["info"]["rec"]
        undef_iou = case["info"]["undef_iou"]
        def_iou = case["def_iou"]
        ub = case["info"]["user_bbox_orig"]
        uw, uh = int(ub[2] - ub[0]), int(ub[3] - ub[1])

        # --- undefended panel ---
        axu = axes[row][0]
        img_u = _overlay_mask(disp_img, case["info"]["undef_mask"], UNDEF_COLOR)
        axu.imshow(img_u)
        _draw_gt_contour(axu, gt_bin)
        _draw_bbox(axu, ub, USER_BOX, f"user bbox {uw}x{uh}")
        axu.set_title(f"{label} CASE - UNDEFENDED\nIoU = {undef_iou:.3f}   "
                      f"(user {rec.user[:6]}..)", fontsize=11)
        axu.axis("off")

        # --- defended panel ---
        axd = axes[row][1]
        img_d = _overlay_mask(disp_img, case["def_mask"], DEF_COLOR)
        axd.imshow(img_d)
        _draw_gt_contour(axd, gt_bin)
        if case["best_bbox"] is not None:
            _draw_bbox(axd, case["best_bbox"], BEST_BOX, "best perturbed bbox")
            extra = ""
        else:
            _draw_bbox(axd, ub, USER_BOX, "user bbox", ls="--")
            extra = f"  (avg over Y, no single box)"
        delta = def_iou - undef_iou
        axd.set_title(f"{label} CASE - DEFENDED ({args.averaging_mode}){extra}\n"
                      f"IoU = {def_iou:.3f}   delta = {delta:+.3f}", fontsize=11)
        axd.axis("off")

    fig.suptitle(
        f"Image '{stem}'  |  defence: {args.averaging_mode}, Y={args.Y}, "
        f"sigma={args.sigma}"
        + (", +base_box" if args.include_base_box else "")
        + "\nyellow = GT contour   red = undefended mask   green = defended mask",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", required=True,
                   help="FOR_TEST root (images/, masks/, user_masks/)")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_name", default="SAM",
                   choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--use", default="mp", help="user_mask kinds: 'm', 'p' or 'mp' (default: both)")
    p.add_argument("--image_stem", default=None,
                   help="which image to visualise (default: first stem alphabetically)")

    # smoothing hyper-parameters (mirror defend_user_study)
    p.add_argument("--Y", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--sigma_center", type=float, default=0.03)
    p.add_argument("--perturb_mode", default="size", choices=["size", "size_center"])
    p.add_argument("--averaging_mode", default="best_of_n",
                   choices=["logit", "sigmoid", "binary", "score_weighted",
                            "best_of_n", "saif_score", "saif_stability"])
    p.add_argument("--include_base_box", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out", default="visualizations/defence_cases.png")
    args = p.parse_args()
    args.use = "".join(sorted(set(args.use.lower())))
    return args


def main():
    args = parse_args()
    from heatmaps.env_dispatch import maybe_dispatch_to_env
    maybe_dispatch_to_env(args.model_name, __file__)
    root = Path(args.root)
    images_dir, masks_dir, user_masks_dir = root / "images", root / "masks", root / "user_masks"

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    records = index_user_masks(user_masks_dir, kinds=tuple(args.use))
    from collections import defaultdict
    per_stem = defaultdict(list)
    for r in records:
        per_stem[r.stem].append(r)
    if not per_stem:
        raise SystemExit("no user masks found")

    stem = args.image_stem or sorted(per_stem.keys())[0]
    if stem not in per_stem:
        raise SystemExit(f"image_stem {stem!r} not found. "
                         f"Available e.g.: {sorted(per_stem)[:5]}")
    print(f"Visualising image '{stem}' ({len(per_stem[stem])} user boxes)")

    predictor = load_model(
        model_name=args.model_name, model_type=args.model_type,
        checkpoint=args.checkpoint_path, device=device,
    )

    disp_img, gt_bin, orig_size, boxes = collect_user_boxes(
        stem, per_stem, images_dir, masks_dir, predictor, device
    )

    worst_info = min(boxes, key=lambda b: b["undef_iou"])
    best_info = max(boxes, key=lambda b: b["undef_iou"])
    print(f"  worst undef_iou = {worst_info['undef_iou']:.3f}   "
          f"best undef_iou = {best_info['undef_iou']:.3f}")

    worst = {"info": worst_info}
    best = {"info": best_info}
    for case in (worst, best):
        m, iou, bb, bi = defend(case["info"], predictor, device, orig_size, args)
        case.update(def_mask=m, def_iou=iou, best_bbox=bb, best_idx=bi)

    render(disp_img, gt_bin, worst, best, args, stem, Path(args.out))


if __name__ == "__main__":
    main()
