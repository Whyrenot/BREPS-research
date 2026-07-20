"""
visualize_single_image.py
=========================
Визуализация для конкретного изображения — ровно 3 колонки на каждый
критический сдвиг:

  Колонка 1 — хороший bbox (best_box) + предсказание SAM
  Колонка 2 — плохой bbox  (bad_box)  + предсказание SAM
  Колонка 3 — Smoothing: все Y перемешанных bbox + только финальная маска

Строка 0 сверху: панель "все bboxes из CSV, раскрашенные по IoU".

Весь инференс делегирован defend_critical_shifts.run_image().
Этот файл НЕ импортирует SAM и не вызывает модель напрямую.

Использование:
    python scripts/visualize_single_image.py \\
        --image_name 21077 \\
        --csv_path computed_test_comp_boxes/SAM/Berkeley/res_final_21077_.csv \\
        --image_path /path/to/Berkeley/images/21077.jpg \\
        --mask_path  /path/to/Berkeley/masks/21077.png \\
        --shifts_json critical_shifts_berkeley.json \\
        --checkpoint_path /path/to/sam_vit_b_01ec64.pth \\
        --out_path visualizations/21077_full_vis.png
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# All SAM inference comes from defend_critical_shifts
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heatmaps.defend_critical_shifts import (
    CaseResult,
    ImageResult,
    boxes_to_original,
    run_image,
)
from heatmaps.comp_hw_smoothed import load_model


# ---------------------------------------------------------------------------
# Pure drawing utilities — no model calls
# ---------------------------------------------------------------------------

def alpha_blend(img_rgb: np.ndarray, mask: np.ndarray,
                color_rgb: tuple, alpha: float) -> np.ndarray:
    out = img_rgb.astype(np.float32)
    overlay = np.zeros_like(out)
    overlay[mask > 0] = color_rgb
    a = alpha * (mask > 0).astype(np.float32)[..., None]
    return np.clip(out * (1 - a) + overlay * a, 0, 255).astype(np.uint8)


def draw_rect(ax, box: np.ndarray, color: str, lw: float = 2.5, alpha: float = 1.0):
    x1, y1, x2, y2 = box
    ax.add_patch(mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha,
    ))


def style_ax(ax, title: str, fontsize: int = 9):
    ax.set_title(title, fontsize=fontsize, color="white", pad=4)
    ax.axis("off")
    ax.set_facecolor("#0d0d1a")


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def panel_all_boxes(ax, image_rgb, gt_mask, df, orig_size, cmap_name="jet"):
    """Top panel: every bbox from the CSV, coloured by IoU."""
    ious = df["iou"].values.astype(np.float32)
    boxes_raw = np.vstack(df["bbox"].apply(
        lambda v: v if isinstance(v, (list, np.ndarray)) else ast.literal_eval(v)
    ).values).astype(np.float32)
    boxes_orig = boxes_to_original(boxes_raw, orig_size)

    vmin, vmax = float(ious.min()), float(ious.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    canvas = alpha_blend(image_rgb, gt_mask, (200, 200, 200), 0.22)
    ax.imshow(canvas)

    for idx in np.argsort(ious):          # worst → best (so best drawn on top)
        x1, y1, x2, y2 = boxes_orig[idx]
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0.5, edgecolor=cmap(norm(ious[idx])),
            facecolor="none", alpha=0.75,
        ))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="IoU")
    style_ax(ax,
             f"All bboxes (coloured by IoU)\n"
             f"n={len(ious)}, min={vmin:.3f}, max={vmax:.3f}")


def panel_best(ax, image_rgb, gt_mask, cr: CaseResult, idx: int):
    """Column 1: best_box + SAM prediction."""
    canvas = alpha_blend(image_rgb, gt_mask, (240, 240, 240), 0.22)
    canvas = alpha_blend(canvas, cr.best_mask, (80, 210, 80), 0.50)
    ax.imshow(canvas)
    draw_rect(ax, cr.best_box_orig, "lime")
    style_ax(ax,
             f"Shift #{idx + 1} | best_box\n"
             f"IoU = {cr.reference_iou:.4f}   Δ = {cr.iou_drop:.4f}")


def panel_bad(ax, image_rgb, gt_mask, cr: CaseResult, idx: int):
    """Column 2: bad_box + SAM prediction."""
    canvas = alpha_blend(image_rgb, gt_mask, (240, 240, 240), 0.22)
    canvas = alpha_blend(canvas, cr.bad_mask, (230, 60, 60), 0.50)
    ax.imshow(canvas)
    draw_rect(ax, cr.bad_box_orig, "red")
    style_ax(ax,
             f"Shift #{idx + 1} | bad_box\n"
             f"IoU = {cr.undefended_iou:.4f}")


def panel_smoothing(ax, image_rgb, gt_mask, cr: CaseResult, idx: int):
    """Column 3: all Y perturbed bboxes + final defended segmentation only."""
    # Defended mask as the main overlay
    canvas = alpha_blend(image_rgb, gt_mask, (240, 240, 240), 0.22)
    canvas = alpha_blend(canvas, cr.defended_mask, (30, 144, 255), 0.52)
    ax.imshow(canvas)

    # Draw all Y perturbed bboxes (thin, semi-transparent yellow)
    for box in cr.perturbed_boxes_orig:
        draw_rect(ax, box, color="yellow", lw=0.6, alpha=0.5)

    # Highlight the original bad_box in orange
    draw_rect(ax, cr.bad_box_orig, "orange", lw=2.0)

    recovery = cr.defended_iou - cr.undefended_iou
    style_ax(ax,
             f"Shift #{idx + 1} | smoothing ({len(cr.perturbed_boxes_orig)} boxes)\n"
             f"IoU = {cr.defended_iou:.4f}   recovery = {recovery:+.4f}")


def panel_final_prediction(ax, image_rgb, gt_mask, img_result: ImageResult):
    """Final SAM prediction using the tight GT bounding box."""
    canvas = alpha_blend(image_rgb, gt_mask, (255, 255, 255), 0.26)
    canvas = alpha_blend(canvas, img_result.final_pred_mask, (30, 144, 255), 0.52)
    ax.imshow(canvas)
    draw_rect(ax, img_result.gt_bbox_orig, "cyan")
    style_ax(ax,
             f"Final prediction (GT bbox)\n"
             f"IoU = {img_result.final_pred_iou:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name",      type=str, required=True)
    parser.add_argument("--csv_path",        type=str, required=True,
                        help="CSV with all bbox IoUs (res_final_<name>_.csv)")
    parser.add_argument("--image_path",      type=str, required=True)
    parser.add_argument("--mask_path",       type=str, required=True)
    parser.add_argument("--shifts_json",     type=str, default="",
                        help="critical_shifts JSON; if empty — no shift rows")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name",      type=str, default="SAM",
                        choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    parser.add_argument("--model_type",      type=str, default="vit_b")
    parser.add_argument("--gpu",             type=int, default=0)
    # smoothing params passed through to run_image
    parser.add_argument("--Y",               type=int,   default=16)
    parser.add_argument("--sigma",           type=float, default=0.05)
    parser.add_argument("--sigma_center",    type=float, default=0.03)
    parser.add_argument("--perturb_mode",    type=str,   default="size",
                        choices=["size", "size_center"])
    parser.add_argument("--averaging_mode",  type=str,   default="sigmoid",
                        choices=["logit", "sigmoid", "binary",
                                 "score_weighted", "best_of_n"])
    parser.add_argument("--out_path",        type=str,
                        default="visualizations/vis_single.png")
    parser.add_argument("--dpi",             type=int, default=150)
    args = parser.parse_args()

    from heatmaps.env_dispatch import maybe_dispatch_to_env
    maybe_dispatch_to_env(args.model_name, __file__)

    # ── load image + mask for drawing ──────────────────────────────────────
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(args.image_path)
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gt_raw = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_raw is None:
        raise FileNotFoundError(args.mask_path)
    gt_mask = (gt_raw > 0).astype(np.uint8)

    # ── CSV for the all-boxes top panel ────────────────────────────────────
    df = pd.read_csv(args.csv_path)
    df["bbox"] = df["bbox"].apply(
        lambda v: v if isinstance(v, (list, np.ndarray)) else ast.literal_eval(v)
    )

    # ── critical shift cases ───────────────────────────────────────────────
    shift_cases: list[dict] = []
    if args.shifts_json:
        with open(args.shifts_json) as f:
            all_shifts = json.load(f)
        shift_cases = [c for c in all_shifts if c["image_name"] == args.image_name]
        print(f"Critical shifts for '{args.image_name}': {len(shift_cases)}")

    # ── load model once, run ALL inference via defend_critical_shifts ───────
    print("Loading model …")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    predictor = load_model(
        model_name=args.model_name,
        model_type=args.model_type,
        checkpoint=args.checkpoint_path,
        device=device,
    )

    print("Running inference (defend_critical_shifts.run_image) …")
    img_result: ImageResult = run_image(
        image_name=args.image_name,
        image_path=args.image_path,
        mask_path=args.mask_path,
        shift_cases=shift_cases,
        predictor=predictor,
        rank=device,
        Y=args.Y,
        sigma=args.sigma,
        sigma_center=args.sigma_center,
        averaging_mode=args.averaging_mode,
        perturb_mode=args.perturb_mode,
    )

    orig_size = img_result.orig_size
    n_shifts  = len(img_result.case_results)

    # Layout:
    #   Row 0  (3 cols): [all-boxes panel (spans 2 cols) | final prediction]
    #   Rows 1..N (3 cols): [best_box | bad_box | smoothing]
    n_rows = 1 + n_shifts
    n_cols = 3

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor("#111122")

    # Row 0: merge first two columns for the all-boxes panel
    ax_all   = fig.add_subplot(n_rows, n_cols, 1)         # col 0
    ax_all2  = fig.add_subplot(n_rows, n_cols, 2)         # col 1 (hidden)
    ax_final = fig.add_subplot(n_rows, n_cols, 3)         # col 2

    # Use gridspec to make the all-boxes panel span 2 columns in row 0
    fig.clf()
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.35, wspace=0.15)

    ax_all   = fig.add_subplot(gs[0, :2])                 # spans cols 0-1
    ax_final = fig.add_subplot(gs[0, 2])

    for ax in [ax_all, ax_final]:
        ax.set_facecolor("#0d0d1a")

    panel_all_boxes(ax_all, image_rgb, gt_mask, df, orig_size)
    panel_final_prediction(ax_final, image_rgb, gt_mask, img_result)

    # Rows 1..N: one per critical shift
    for i, cr in enumerate(img_result.case_results):
        print(f"  Shift {i+1}/{n_shifts}: "
              f"drop={cr.iou_drop:.4f}  "
              f"best={cr.reference_iou:.4f}  "
              f"bad={cr.undefended_iou:.4f}  "
              f"defended={cr.defended_iou:.4f}")
        row = i + 1
        ax_b  = fig.add_subplot(gs[row, 0])
        ax_bd = fig.add_subplot(gs[row, 1])
        ax_sm = fig.add_subplot(gs[row, 2])
        for ax in [ax_b, ax_bd, ax_sm]:
            ax.set_facecolor("#0d0d1a")

        panel_best(ax_b,  image_rgb, gt_mask, cr, i)
        panel_bad(ax_bd,  image_rgb, gt_mask, cr, i)
        panel_smoothing(ax_sm, image_rgb, gt_mask, cr, i)

    fig.suptitle(
        f"Image: {args.image_name}   |   Critical shifts: {n_shifts}",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
