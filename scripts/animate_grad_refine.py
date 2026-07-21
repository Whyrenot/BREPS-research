"""
animate_grad_refine.py
======================
Animate the gradient box-refinement on ONE user_study image: pick the WEAKEST
user box (lowest undefended IoU) for a given image, run gradient ascent on SAM's
predicted-IoU, and render a GIF where each frame shows
  - left  : the image with the current box + a semi-transparent overlay of
            pred mask vs GT  (yellow = intersection / TP, red = FP, blue = FN)
  - right : predicted-IoU and true-IoU curves vs gradient step (moving marker)

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/animate_grad_refine.py \
        --root ../user_study/FOR_TEST --stem Berkeley_42078 \
        --checkpoint_path /.../sam_vit_b_01ec64.pth \
        --model_name SAM --model_type vit_b \
        --use mp --steps 50 --lr 1 --multimask \
        --out visualizations/grad_anim_Berkeley_42078.gif --fps 6
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for sibling import

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from heatmaps.comp_hw_smoothed import batch_iou_torch, get_bbox_from_mask, get_original_size, load_model
from heatmaps.defend_critical_shifts import (
    _predict_single_box, _prepare_image, boxes_to_original, original_to_1024,
)
from heatmaps.defend_user_study import _load_binary_mask, index_user_masks
from refine_box_iou_grad import refine_box_by_iou_grad


def _load_display_image(image_path, H, W):
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H))
    return img


def _mask_for(box_1024, head, multimask, predictor, device):
    """Binary mask (H,W) for a box (1024 frame) at the given head."""
    box_t = torch.as_tensor(box_1024, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.inference_mode():
        masks, _, _ = predictor.predict_torch(None, None, boxes=box_t,
                                              multimask_output=multimask,
                                              return_logits=False)
    return masks[0, head].cpu().numpy().astype(bool)


def _overlay(disp, pred_mask, gt):
    """pred vs GT overlay: yellow=intersection(TP), red=FP, blue=FN (missed)."""
    out = disp.astype(np.float32).copy()
    inter = pred_mask & gt
    fp = pred_mask & (~gt)
    fn = (~pred_mask) & gt

    def paint(m, color, a):
        if m.any():
            out[m] = (1 - a) * out[m] + a * np.asarray(color, np.float32)

    paint(fp, (220, 50, 50), 0.30)
    paint(fn, (60, 120, 220), 0.30)
    paint(inter, (255, 220, 30), 0.55)
    return np.clip(out, 0, 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="user_study FOR_TEST root")
    p.add_argument("--stem", required=True, help="image stem")
    p.add_argument("--use", default="mp", help="user mask kinds: m/p/mp (default: both)")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_name", default="SAM",
                   choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--multimask", action="store_true", default=False)

    p.add_argument("--out", default="visualizations/grad_anim.gif")
    p.add_argument("--fps", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    from heatmaps.env_dispatch import maybe_dispatch_to_env
    maybe_dispatch_to_env(args.model_name, __file__)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    image_path = Path(args.root) / "images" / f"{args.stem}.png"
    mask_path = Path(args.root) / "masks" / f"{args.stem}.png"
    user_masks_dir = Path(args.root) / "user_masks"

    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=device)
    for prm in predictor.model.parameters():
        prm.requires_grad_(False)

    _prepare_image(str(image_path), predictor)
    H, W = get_original_size(predictor)
    disp = _load_display_image(image_path, H, W)
    gt = _load_binary_mask(mask_path, target_shape=(H, W))
    gt_t = torch.from_numpy(gt)

    def iou_vs_gt(mask_bool):
        return batch_iou_torch(gt_t.unsqueeze(0).unsqueeze(0),
                               torch.from_numpy(mask_bool).unsqueeze(0).unsqueeze(0)).item()

    # ---- pick the WEAKEST user box for this image ----
    use = "".join(sorted(set(args.use.lower())))
    recs = [r for r in index_user_masks(user_masks_dir, kinds=tuple(use)) if r.stem == args.stem]
    if not recs:
        raise SystemExit(f"no user masks for stem {args.stem}")

    weakest = None
    for r in recs:
        um = _load_binary_mask(r.path, target_shape=(H, W))
        if um.sum() == 0:
            continue
        rmin, rmax, cmin, cmax = get_bbox_from_mask(um.astype(np.uint8))
        ub = np.array([cmin, rmin, cmax, rmax], dtype=np.int32)
        if ub[2] - ub[0] <= 0 or ub[3] - ub[1] <= 0:
            continue
        box1024 = original_to_1024(ub[None], (H, W))[0]
        m0 = _predict_single_box(torch.tensor(box1024, dtype=torch.float32), predictor,
                                 device, boxes_already_transformed=True).numpy()
        iou = iou_vs_gt(m0)
        if weakest is None or iou < weakest["iou"]:
            weakest = {"box1024": box1024, "iou": iou, "user": r.user}
    if weakest is None:
        raise SystemExit("no usable user box")
    print(f"weakest user box: undef IoU={weakest['iou']:.3f} (user {weakest['user'][:6]})")

    # ---- gradient trajectory ----
    _final, traj = refine_box_by_iou_grad(
        weakest["box1024"], predictor, device, steps=args.steps, lr=args.lr,
        multimask=args.multimask, gt_tensor=gt_t)

    # ---- precompute frames ----
    frames, boxes_orig, preds, trues = [], [], [], []
    for t in traj:
        mask = _mask_for(t["box"], t["head"], args.multimask, predictor, device)
        frames.append(_overlay(disp, mask, gt))
        boxes_orig.append(boxes_to_original(np.array([t["box"]]), (H, W))[0])
        preds.append(t["pred_score"]); trues.append(t["true_iou"])
    steps = [t["step"] for t in traj]
    print(f"trajectory: true {trues[0]:.3f} -> {trues[-1]:.3f} (best {max(trues):.3f}); "
          f"pred {preds[0]:.3f} -> {preds[-1]:.3f}")

    _animate(frames, boxes_orig, steps, preds, trues, args.stem, weakest["iou"],
             Path(args.out), args.fps)


def _animate(frames, boxes_orig, steps, preds, trues, stem, undef_iou, out_path, fps):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    imobj = axl.imshow(frames[0])
    x0, y0, x1, y1 = boxes_orig[0]
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", lw=2.5)
    axl.add_patch(rect)
    axl.axis("off")
    axl.set_title("", fontsize=12)

    axr.set_xlim(steps[0], steps[-1]); axr.set_ylim(0, 1)
    axr.set_xlabel("gradient step"); axr.set_ylabel("IoU")
    axr.grid(ls=":", alpha=0.4)
    axr.axhline(undef_iou, color="gray", ls="--", lw=1, label=f"start true ({undef_iou:.3f})")
    (line_pred,) = axr.plot([], [], "-o", color="#9467bd", ms=3, label="predicted IoU")
    (line_true,) = axr.plot([], [], "-o", color="#1f77b4", ms=3, label="true IoU")
    vline = axr.axvline(steps[0], color="k", lw=0.8, alpha=0.5)
    axr.legend(loc="lower right")
    axr.set_title("predicted vs true IoU per step")

    # yellow=TP(intersection) red=FP blue=FN  legend (text)
    axl.text(0.01, 0.99, "yellow = pred∩GT   red = FP   blue = missed GT",
             transform=axl.transAxes, va="top", ha="left", fontsize=9,
             color="white", bbox=dict(facecolor="black", alpha=0.5, pad=2))

    def update(k):
        imobj.set_data(frames[k])
        bx0, by0, bx1, by1 = boxes_orig[k]
        rect.set_bounds(bx0, by0, bx1 - bx0, by1 - by0)
        line_pred.set_data(steps[:k + 1], preds[:k + 1])
        line_true.set_data(steps[:k + 1], trues[:k + 1])
        vline.set_xdata([steps[k], steps[k]])
        axl.set_title(f"{stem}  weakest box  |  step {steps[k]}/{steps[-1]}   "
                      f"pred={preds[k]:.3f}   true={trues[k]:.3f}", fontsize=12)
        return imobj, rect, line_pred, line_true, vline

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved animation -> {out_path}  ({len(frames)} frames, {fps} fps)")


if __name__ == "__main__":
    main()
