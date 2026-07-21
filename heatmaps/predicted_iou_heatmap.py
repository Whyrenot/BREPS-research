"""
predicted_iou_heatmap.py
========================
Visualise WHAT the gradient method optimises toward: SAM's predicted-IoU head
as a function of the box prompt, on the SAME (width x height) grid used for the
real IoU heatmaps (comp_hw.generate_bounding_boxes) -- box centre fixed at the
GT-mask centre, axes = box width and height.

For every (w, h) box we run SAM and read all FOUR mask tokens:
    head 0      = the single-output token (multimask_output=False)
    heads 1..3  = the multimask tokens   (multimask_output=True)
We record, per head, BOTH:
    - predicted IoU  (the IoU-prediction head -- what we maximise)
    - true IoU vs GT (reality)
plus the argmax-over-heads map (which head SAM is most confident in) and the
true IoU you actually obtain at that argmax head.

Output figure (2 rows x 5 cols):
    row PRED : head0 | head1 | head2 | head3 | argmax-head
    row TRUE : head0 | head1 | head2 | head3 | true@argmax
Top row = "to what we optimise"; bottom row = "what you actually get".

Example (user_study FOR_TEST):  
    CUDA_VISIBLE_DEVICES=3 python heatmaps/predicted_iou_heatmap.py \
        --root ../user_study/FOR_TEST --stem 100080 \
        --checkpoint_path /.../sam_vit_b_01ec64.pth \
        --model_name SAM --model_type vit_b \
        --grid_step 16 \
        --out visualizations/pred_iou_heatmap_100080.png
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from heatmaps.comp_hw_smoothed import batch_iou_torch, get_bbox_from_mask, get_original_size, load_model
from heatmaps.defend_critical_shifts import _prepare_image

_HEAD_COLORS = np.array([[31, 119, 180], [255, 127, 14],
                         [44, 160, 44], [214, 39, 40]], dtype=np.uint8)  # heads 0..3


def _load_display_image(image_path, H, W) -> np.ndarray:
    """RGB image at the working resolution (H, W) = predictor.original_size."""
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H))
    return img


def _splat(boxes_int, values, H, W, block):
    """Paint each box's 4 corners (block x block tiles) with its value; return
    (mean canvas, coverage mask). Mirrors draw_heatmaps' corner convention but
    tiled so the subsampled grid leaves no gaps."""
    acc = np.zeros((H, W), np.float64)
    cnt = np.zeros((H, W), np.float64)
    half = max(1, block) // 2
    for (x0, y0, x1, y1), v in zip(boxes_int, values):
        if not np.isfinite(v):
            continue
        for px, py in ((x0, y0), (x1, y1), (x0, y1), (x1, y0)):
            xa, xb = max(0, px - half), min(W, px + half + 1)
            ya, yb = max(0, py - half), min(H, py + half + 1)
            acc[ya:yb, xa:xb] += v
            cnt[ya:yb, xa:xb] += 1
    cover = cnt > 0
    canvas = np.zeros((H, W), np.float32)
    canvas[cover] = (acc[cover] / cnt[cover]).astype(np.float32)
    return canvas, cover


def _splat_categorical(boxes_int, head_idx, H, W, block):
    """Last-write splat of a categorical head index (0..3). Returns (idx, cover)."""
    idx = np.full((H, W), -1, np.int32)
    half = max(1, block) // 2
    for (x0, y0, x1, y1), hd in zip(boxes_int, head_idx):
        for px, py in ((x0, y0), (x1, y1), (x0, y1), (x1, y0)):
            xa, xb = max(0, px - half), min(W, px + half + 1)
            ya, yb = max(0, py - half), min(H, py + half + 1)
            idx[ya:yb, xa:xb] = hd
    return idx, idx >= 0


def _blend(disp, rgb, cover, alpha):
    out = disp.astype(np.float32).copy()
    out[cover] = (1 - alpha) * disp[cover] + alpha * rgb[cover].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_gt(mask_path, predictor) -> np.ndarray:
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(mask_path)
    gt = (gt > 0).astype(np.uint8)
    H, W = get_original_size(predictor)
    if gt.shape != (H, W):
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    return (gt > 0)


def _predict_all_heads(boxes_orig, predictor, device):
    """boxes_orig: (B,4) float in original image coords. Returns
    masks (B,4,H,W) bool, scores (B,4): token-0 then 3 multimask heads."""
    tb = torch.as_tensor(boxes_orig, dtype=torch.float32)
    if hasattr(predictor, "transform"):
        tb = predictor.transform.apply_boxes_torch(tb, predictor.original_size)
    tb = tb.to(device)
    with torch.inference_mode():
        m0, s0, _ = predictor.predict_torch(None, None, boxes=tb,
                                            multimask_output=False, return_logits=False)
        mm, sm, _ = predictor.predict_torch(None, None, boxes=tb,
                                            multimask_output=True, return_logits=False)
    masks = torch.cat([m0, mm], dim=1)   # (B,4,H,W)
    scores = torch.cat([s0, sm], dim=1)  # (B,4)
    return masks, scores


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # source: root+stem (preferred) or explicit paths
    p.add_argument("--root", default=None, help="user_study FOR_TEST root")
    p.add_argument("--stem", default=None, help="image stem (with --root)")
    p.add_argument("--image_path", default=None)
    p.add_argument("--mask_path", default=None)

    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_name", default="SAM",
                   choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--gpu", type=int, default=0)

    # (width x height) grid, centre fixed at GT centre (comp_hw convention)
    p.add_argument("--grid_step", type=int, default=16, help="step (px) of the w/h grid")
    p.add_argument("--max_w", type=int, default=0, help="max box width (0 = image width)")
    p.add_argument("--max_h", type=int, default=0, help="max box height (0 = image height)")
    p.add_argument("--batch_size", type=int, default=48)

    p.add_argument("--out", default="visualizations/pred_iou_heatmap.png",
                   help="(width x height) panel figure")
    p.add_argument("--overlay_out", default=None,
                   help="image-space overlay figure (default: <out>_overlay.png)")
    p.add_argument("--overlay_alpha", type=float, default=0.55)
    p.add_argument("--save_npz", default=None, help="optional .npz with all raw grids")
    return p.parse_args()


def main():
    args = parse_args()
    from heatmaps.env_dispatch import maybe_dispatch_to_env
    maybe_dispatch_to_env(args.model_name, __file__)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.root and args.stem:
        image_path = Path(args.root) / "images" / f"{args.stem}.png"
        mask_path = Path(args.root) / "masks" / f"{args.stem}.png"
        stem = args.stem
    elif args.image_path and args.mask_path:
        image_path, mask_path = Path(args.image_path), Path(args.mask_path)
        stem = image_path.stem
    else:
        raise SystemExit("Provide --root+--stem or --image_path+--mask_path")

    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=device)
    _prepare_image(str(image_path), predictor)          # sets predictor's original-size state
    H, W = get_original_size(predictor)
    gt = _load_gt(mask_path, predictor)
    gt_t = torch.from_numpy(gt).to(device)
    if gt.sum() == 0:
        raise SystemExit("empty GT mask")

    # GT tight bbox -> centre & reference (w*, h*)
    rmin, rmax, cmin, cmax = get_bbox_from_mask(gt.astype(np.uint8))
    cx = (cmin + cmax) // 2
    cy = (rmin + rmax) // 2
    w_star, h_star = int(cmax - cmin), int(rmax - rmin)

    max_w = args.max_w or W
    max_h = args.max_h or H
    ws = np.arange(args.grid_step, max_w + 1, args.grid_step, dtype=np.int32)
    hs = np.arange(args.grid_step, max_h + 1, args.grid_step, dtype=np.int32)
    print(f"image '{stem}' {W}x{H}  | GT centre=({cx},{cy})  GT size=({w_star}x{h_star})")
    print(f"grid: {len(ws)} widths x {len(hs)} heights = {len(ws)*len(hs)} boxes, step={args.grid_step}")

    # build all (w,h) boxes (centre fixed at GT centre), clipped like comp_hw
    boxes, idx = [], []
    for ih, h in enumerate(hs):
        for iw, w in enumerate(ws):
            x0 = np.clip(cx - w / 2, 0, W - 1)
            x1 = np.clip(cx + w / 2, 0, W - 1)
            y0 = np.clip(cy - h / 2, 0, H - 1)
            y1 = np.clip(cy + h / 2, 0, H - 1)
            boxes.append([x0, y0, x1, y1])
            idx.append((ih, iw))
    boxes = np.asarray(boxes, dtype=np.float32)

    n_heads = 4
    pred = np.full((len(hs), len(ws), n_heads), np.nan, dtype=np.float32)
    true = np.full((len(hs), len(ws), n_heads), np.nan, dtype=np.float32)

    from tqdm import tqdm
    for s in tqdm(range(0, len(boxes), args.batch_size), desc="grid"):
        bb = boxes[s:s + args.batch_size]
        masks, scores = _predict_all_heads(bb, predictor, device)  # (B,4,H,W),(B,4)
        B = masks.shape[0]
        gt_rep = gt_t[None, None].expand(B, 1, H, W)
        for hd in range(n_heads):
            iou = batch_iou_torch(gt_rep, masks[:, hd:hd + 1]).cpu().numpy()
            sc = scores[:, hd].cpu().numpy()
            for j in range(B):
                ih, iw = idx[s + j]
                true[ih, iw, hd] = iou[j]
                pred[ih, iw, hd] = sc[j]

    argmax_head = np.argmax(pred, axis=2)                       # (Hg,Wg) in 0..3
    ii, jj = np.meshgrid(np.arange(len(hs)), np.arange(len(ws)), indexing="ij")
    true_at_argmax = true[ii, jj, argmax_head]
    max_pred = np.nanmax(pred, axis=2)   # the OBJECTIVE the gradient climbs
    max_true = np.nanmax(true, axis=2)   # oracle: best achievable head (true)

    if args.save_npz:
        Path(args.save_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_npz, ws=ws, hs=hs, pred=pred, true=true,
                 argmax_head=argmax_head, true_at_argmax=true_at_argmax,
                 max_pred=max_pred, max_true=max_true, gt_w=w_star, gt_h=h_star)

    _plot(ws, hs, pred, true, argmax_head, true_at_argmax, max_pred, max_true,
          w_star, h_star, stem, Path(args.out))

    # image-space overlay (corner-splat over the photo, draw_heatmaps style)
    disp = _load_display_image(image_path, H, W)
    boxes_int = np.round(boxes).astype(np.int32)
    overlay_out = Path(args.overlay_out) if args.overlay_out else \
        Path(args.out).with_name(Path(args.out).stem + "_overlay.png")
    _plot_overlay(disp, boxes_int, pred, true, argmax_head, true_at_argmax,
                  max_pred, max_true, args.grid_step, args.overlay_alpha,
                  (cx, cy), stem, overlay_out)

    # quick text: where predicted peaks vs where true peaks (head 0)
    def _peak(grid):
        k = np.nanargmax(grid); ih, iw = np.unravel_index(k, grid.shape)
        return ws[iw], hs[ih], float(grid[ih, iw])
    pw, ph, pv = _peak(pred[:, :, 0]); tw, th, tv = _peak(true[:, :, 0])
    print(f"head0: predicted peak @ (w={pw},h={ph}) score={pv:.3f}  |  "
          f"true peak @ (w={tw},h={th}) IoU={tv:.3f}  |  GT box=({w_star}x{h_star})")


def _plot(ws, hs, pred, true, argmax_head, true_at_argmax, max_pred, max_true,
          w_star, h_star, stem, out_path):
    extent = [ws[0], ws[-1], hs[0], hs[-1]]
    fig, axes = plt.subplots(2, 6, figsize=(26, 9), constrained_layout=True)
    cont = []  # continuous panels -> shared colorbar

    def _hm(ax, grid, title, cmap="viridis", vmin=0.0, vmax=1.0):
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot([w_star], [h_star], marker="*", color="red", ms=14,
                markeredgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("box width"); ax.set_ylabel("box height")
        cont.append(ax)
        return im

    head_titles = ["head 0 (single-out)", "head 1", "head 2", "head 3"]
    for hd in range(4):
        im = _hm(axes[0][hd], pred[:, :, hd], f"PRED IoU — {head_titles[hd]}")
        _hm(axes[1][hd], true[:, :, hd], f"TRUE IoU — {head_titles[hd]}")
    # col 4: the OBJECTIVE (max predicted over heads) vs reality there
    im = _hm(axes[0][4], max_pred, "PRED max over heads (OBJECTIVE)")
    _hm(axes[1][4], true_at_argmax, "TRUE IoU @ argmax head")
    # col 5 bottom: oracle (best head by true IoU)
    _hm(axes[1][5], max_true, "TRUE max over heads (oracle)")
    fig.colorbar(im, ax=cont, shrink=0.5, label="IoU")

    # col 5 top: argmax-head map (categorical)
    cmap4 = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axc = axes[0][5]
    imc = axc.imshow(argmax_head, origin="lower", extent=extent, aspect="auto",
                     cmap=cmap4, vmin=-0.5, vmax=3.5)
    axc.plot([w_star], [h_star], marker="*", color="white", ms=14, markeredgecolor="black")
    axc.set_title("argmax head (which head wins)", fontsize=10)
    axc.set_xlabel("box width"); axc.set_ylabel("box height")
    cb = fig.colorbar(imc, ax=axc, ticks=[0, 1, 2, 3], shrink=0.8)
    cb.set_label("chosen head")

    fig.suptitle(
        f"SAM predicted vs true IoU over (width x height), centre = GT centre — '{stem}'\n"
        f"top = what we optimise (predicted-IoU); 'PRED max over heads' = the actual objective; "
        f"bottom = reality (true IoU); red star = GT box size",
        fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def _plot_overlay(disp, boxes_int, pred, true, argmax_head, true_at_argmax,
                  max_pred, max_true, block, alpha, center, stem, out_path):
    """Same 2x6 layout but each panel = the photo with the corner-splat heatmap
    overlaid (box corners in image space, jet colormap)."""
    H, W = disp.shape[:2]
    cx, cy = center
    jet = plt.get_cmap("jet")

    def panel(ax, values_grid, title, categorical=False):
        vals = values_grid.reshape(-1)
        if categorical:
            idx, cover = _splat_categorical(boxes_int, vals.astype(int), H, W, block)
            rgb = np.zeros((H, W, 3), np.uint8)
            for h in range(4):
                rgb[idx == h] = _HEAD_COLORS[h]
            blended = _blend(disp, rgb, cover, alpha)
        else:
            canvas, cover = _splat(boxes_int, vals, H, W, block)
            rgb = (jet(np.clip(canvas, 0, 1))[..., :3] * 255).astype(np.uint8)
            blended = _blend(disp, rgb, cover, alpha)
        ax.imshow(blended)
        ax.plot([cx], [cy], marker="+", color="white", ms=12, mew=2)
        ax.set_title(title, fontsize=10); ax.axis("off")

    fig, axes = plt.subplots(2, 6, figsize=(28, 10), constrained_layout=True)
    head_titles = ["head 0 (single-out)", "head 1", "head 2", "head 3"]
    for hd in range(4):
        panel(axes[0][hd], pred[:, :, hd], f"PRED IoU — {head_titles[hd]}")
        panel(axes[1][hd], true[:, :, hd], f"TRUE IoU — {head_titles[hd]}")
    panel(axes[0][4], max_pred, "PRED max over heads (OBJECTIVE)")
    panel(axes[1][4], true_at_argmax, "TRUE IoU @ argmax head")
    panel(axes[0][5], argmax_head.astype(np.float32),
          "argmax head (blue/orange/green/red = 0/1/2/3)", categorical=True)
    panel(axes[1][5], max_true, "TRUE max over heads (oracle)")

    # jet colorbar for the continuous panels
    sm = cm.ScalarMappable(cmap=jet); sm.set_array([0, 1])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, label="IoU")

    fig.suptitle(
        f"Predicted vs true IoU overlaid on image (box corners, centre fixed at GT centre) — '{stem}'\n"
        f"each pixel = IoU when a box CORNER lands there; white + = GT centre",
        fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay -> {out_path}")


if __name__ == "__main__":
    main()
