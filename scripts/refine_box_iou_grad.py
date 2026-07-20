"""
refine_box_iou_grad.py
======================
Test-time box refinement by GRADIENT ASCENT on SAM's own predicted-IoU.

Motivation: best_of_n (pick the perturbed box with the highest predicted IoU)
beats stability-based selection, i.e. SAM's predicted-IoU head is an informative
signal. best_of_n exploits it by *random search*; here we exploit it by
*gradient ascent* -- the box (cx, cy, w, h) is a continuous, differentiable
input, so we can compute d(predicted_iou)/d(box) and step the box uphill.

The box -> prompt_encoder -> mask_decoder -> iou_prediction_head path is fully
differentiable (corner coords go through PositionEmbeddingRandom: normalise ->
Gaussian projection -> sin/cos). The image embedding is cached by set_image and
does NOT depend on the box, so each step is one cheap decoder forward/backward
(the heavy image encoder is untouched).

This script runs, per critical-shift case (starting from the BROKEN bad_box):
    - undefended  : true IoU of SAM(bad_box)
    - best_of_n   : true IoU of the smoothing defence (for comparison)
    - grad-refine : true & predicted IoU at every gradient step
and plots true-IoU-vs-step (with undefended / best_of_n reference lines) plus a
predicted-vs-true calibration curve -- the key sanity check: does pushing the
*predicted* IoU up actually pull the *true* IoU up?

It also dumps, over ALL processed cases, the full (GT IoU, predicted IoU)
vectors (--calib_csv), scatters one against the other and reports the Pearson /
Spearman correlation between the two vectors (--calib_plot), for both the
undefended start box and the final gradient-refined box.

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/refine_box_iou_grad.py \
        --critical_shifts critical_shifts_coco.json \
        --images_dir /.../COCO_MVal/img \
        --masks_dir  /.../COCO_MVal/gt \
        --checkpoint_path /.../sam_vit_b_01ec64.pth \
        --model_name SAM --model_type vit_b \
        --limit 100 --steps 20 --lr 3.0 --multimask \
        --out_csv results/grad_refine_percase.csv \
        --out_plot results/grad_refine.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heatmaps.comp_hw_smoothed import (
    batch_iou_torch,
    get_bbox_from_mask,
    load_model,
    sample_size_and_center_perturbed_boxes,
    sample_size_perturbed_boxes,
)
from heatmaps.defend_critical_shifts import (
    _find_file,
    _predict_single_box,
    _predict_smoothed_box,
    _prepare_image,
    original_to_1024,
)
from heatmaps.defend_user_study import _load_binary_mask, index_user_masks
from heatmaps.env_dispatch import maybe_dispatch_to_env


# ---------------------------------------------------------------------------
# the prototype: gradient ascent on predicted IoU
# ---------------------------------------------------------------------------

def refine_box_by_iou_grad(
    box_1024,
    predictor,
    device,
    steps: int = 20,
    lr: float = 3.0,
    multimask: bool = True,
    gt_tensor: torch.Tensor | None = None,
):
    """Ascend SAM's predicted-IoU w.r.t. the box (cx, cy, w, h).

    box_1024 : length-4 (x0,y0,x1,y1) box in SAM's 1024 input frame.
    gt_tensor: optional GT bool mask (H,W) on CPU -> records true IoU per step.

    Returns (refined_box_1024 tensor, trajectory list[dict]).
    Trajectory entry: {step, pred_score, [true_iou], [box]}.
    """
    model = predictor.model
    image_pe = model.prompt_encoder.get_dense_pe()
    feats = predictor.features
    thr = model.mask_threshold

    b = torch.as_tensor(box_1024, dtype=torch.float32, device=device)
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    w = (b[2] - b[0]).clamp(min=2.0)
    h = (b[3] - b[1]).clamp(min=2.0)
    params = torch.tensor([cx, cy, w, h], device=device, requires_grad=True)
    opt = torch.optim.Adam([params], lr=lr)

    def params_to_box(p):
        ww = p[2].clamp(min=2.0)
        hh = p[3].clamp(min=2.0)
        return torch.stack([p[0] - ww / 2, p[1] - hh / 2, p[0] + ww / 2, p[1] + hh / 2])

    def true_iou_of(low_res, head):
        with torch.no_grad():
            full = model.postprocess_masks(
                low_res[:, head:head + 1].detach(),
                predictor.input_size, predictor.original_size,
            )
            mbin = (full[0, 0] > thr).cpu()
        return batch_iou_torch(
            gt_tensor.unsqueeze(0).unsqueeze(0), mbin.unsqueeze(0).unsqueeze(0)
        ).item()

    traj: list[dict] = []
    with torch.enable_grad():
        for step in range(steps + 1):
            box = params_to_box(params)
            sparse, dense = model.prompt_encoder(points=None, boxes=box[None, :], masks=None)
            low_res, iou_pred = model.mask_decoder(
                image_embeddings=feats,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=multimask,
            )
            head = int(torch.argmax(iou_pred[0]).item()) if multimask else 0
            score = iou_pred[0, head]

            rec = {"step": step, "pred_score": float(score.item()), "head": head}
            if gt_tensor is not None:
                rec["true_iou"] = true_iou_of(low_res, head)
                rec["box"] = box.detach().cpu().numpy().tolist()
            traj.append(rec)

            if step == steps:
                break
            opt.zero_grad()
            (-score).backward()
            opt.step()

    return params_to_box(params).detach(), traj


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------

def _load_gt(mask_path, predictor) -> torch.Tensor:
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(mask_path)
    gt = (gt > 0).astype(np.uint8)
    H, W = predictor.original_size
    if gt.shape != (H, W):
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(gt > 0)


def _iou(gt_tensor, pred_bool_t) -> float:
    return batch_iou_torch(
        gt_tensor.unsqueeze(0).unsqueeze(0),
        pred_bool_t.cpu().unsqueeze(0).unsqueeze(0),
    ).item()


def _box_metrics(box_new, box_ref) -> dict:
    """Displacement of box_new relative to box_ref (both [x0,y0,x1,y1], 1024 px)."""
    b1 = np.asarray(box_new, dtype=np.float64)
    b0 = np.asarray(box_ref, dtype=np.float64)
    c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
    c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
    return {
        "corner_l2": float(np.linalg.norm(b1 - b0)),
        "center_shift": float(np.linalg.norm(c1 - c0)),
        "w_delta": float((b1[2] - b1[0]) - (b0[2] - b0[0])),
        "h_delta": float((b1[3] - b1[1]) - (b0[3] - b0[1])),
    }


def best_of_n_multimask(bad_box, predictor, device, Y, sigma, sigma_center, perturb_mode, seed):
    """best_of_n that searches over BOTH perturbed boxes AND the 3 multimask
    heads: pick the (box, head) with the highest predicted IoU.

    Returns (best_mask bool (H,W), pred_score, chosen_box_1024 np, head_idx).
    """
    sam_h, sam_w = getattr(predictor, "input_size", (1024, 1024))
    base = torch.as_tensor(bad_box, dtype=torch.float32)
    if perturb_mode == "size_center":
        perturbed = sample_size_and_center_perturbed_boxes(
            base, (sam_h, sam_w), Y, sigma, sigma, sigma_center, sigma_center, seed)
    else:
        perturbed = sample_size_perturbed_boxes(base, (sam_h, sam_w), Y, sigma, sigma, seed)

    perturbed_t = perturbed.float().to(device)
    with torch.no_grad():
        try:
            masks, scores, _ = predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=perturbed_t,
                multimask_output=True, return_logits=False)
        except TypeError:
            masks, scores, _ = predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=perturbed_t,
                multimask_output=True)
            masks = masks > predictor.model.mask_threshold

    n_heads = scores.shape[1]
    flat = int(torch.argmax(scores.reshape(-1)).item())
    yi, hi = flat // n_heads, flat % n_heads
    return (masks[yi, hi].cpu(), float(scores[yi, hi].item()),
            perturbed[yi].cpu().numpy(), int(hi))


# ---------------------------------------------------------------------------
# SAIF-style stability  (methods A / B / C)
# ---------------------------------------------------------------------------

def _predict_boxes(boxes_t, predictor, multimask):
    with torch.no_grad():
        try:
            masks, scores, _ = predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=boxes_t,
                multimask_output=multimask, return_logits=False)
        except TypeError:
            masks, scores, _ = predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=boxes_t,
                multimask_output=multimask)
            masks = masks > predictor.model.mask_threshold
    return masks, scores


def _pair_iou(a, b) -> float:
    a = a.bool(); b = b.bool()
    inter = (a & b).sum().item(); union = (a | b).sum().item()
    return inter / union if union > 0 else 1.0


def stability_score(box_np, predictor, device, use_mm, head, M, sigma_s, seed):
    """SAIF-style consistency: mean IoU of the candidate mask under M small box
    perturbations. use_mm/head identify the token (use_mm=False -> token-0;
    use_mm=True -> multimask head 0..2). Returns (stability in [0,1], base_mask)."""
    sam_h, sam_w = getattr(predictor, "input_size", (1024, 1024))
    base = torch.as_tensor(box_np, dtype=torch.float32)
    pert = sample_size_and_center_perturbed_boxes(
        base, (sam_h, sam_w), M, sigma_s, sigma_s, sigma_s, sigma_s, seed)
    allb = torch.cat([base.unsqueeze(0), pert.float()], dim=0).to(device)
    masks, _ = _predict_boxes(allb, predictor, use_mm)
    cand = masks[:, head]                       # (M+1, H, W)
    base_mask = cand[0]
    ious = [_pair_iou(base_mask, cand[j]) for j in range(1, cand.shape[0])]
    return (float(np.mean(ious)) if ious else 1.0), base_mask.cpu()


def method_stab_select(bad_box, predictor, device, Y, sigma, sigma_center,
                       perturb_mode, seed, K, M, sigma_s):
    """(B) best_of_n over Y boxes x ALL 4 tokens (token-0 + 3 multimask heads):
    take top-K by predicted IoU, then re-rank by STABILITY (not predicted).

    Including token-0 is the fix for the good-case regression of best_of_n x mm
    (which used heads 1-3 only and dropped the well-calibrated single-output token).
    Returns (mask, stability, box, token_idx 0..3, pred).
    """
    sam_h, sam_w = getattr(predictor, "input_size", (1024, 1024))
    base = torch.as_tensor(bad_box, dtype=torch.float32)
    if perturb_mode == "size_center":
        perturbed = sample_size_and_center_perturbed_boxes(
            base, (sam_h, sam_w), Y, sigma, sigma, sigma_center, sigma_center, seed)
    else:
        perturbed = sample_size_perturbed_boxes(base, (sam_h, sam_w), Y, sigma, sigma, seed)
    pt = perturbed.float().to(device)

    m0, s0 = _predict_boxes(pt, predictor, False)   # token-0  (Y,1,*),(Y,1)
    mm, sm = _predict_boxes(pt, predictor, True)     # heads1-3 (Y,3,*),(Y,3)
    masks = torch.cat([m0, mm], dim=1)               # (Y,4,*)  idx 0=token0,1..3=mm heads
    scores = torch.cat([s0, sm], dim=1)              # (Y,4)
    nh = scores.shape[1]

    flat = scores.reshape(-1)
    topk = torch.topk(flat, min(K, flat.numel())).indices.tolist()
    best = None
    for idx in topk:
        yi, hi = idx // nh, idx % nh
        use_mm = hi >= 1
        head = (hi - 1) if use_mm else 0
        boxnp = perturbed[yi].cpu().numpy()
        stab, _ = stability_score(boxnp, predictor, device, use_mm, head, M, sigma_s, seed + 1)
        if best is None or stab > best[0]:
            best = (stab, masks[yi, hi].cpu(), boxnp, int(hi), float(scores[yi, hi]))
    return best[1], best[0], best[2], best[3], best[4]


def _pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y) -> float:
    """Spearman rho = Pearson on (tie-aware) ranks."""
    return _pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def _assign_deciles(
    df: "pd.DataFrame",
    image_col: str,
    rank_col: str,
    n_groups: int,
) -> "pd.Series":
    """Per image: sort rows by rank_col ascending, split into n_groups
    equal-ish chunks (np.array_split), 1-indexed (group 1 = worst rank_col,
    n_groups = best). Same convention as scripts/group_robustness_deciles.py
    -- group membership is FIXED by rank_col, so it can be reused to slice
    any other column (predicted/true IoU of clean/attacked/defended boxes).

    Rows with NaN rank_col are left as NaN (excluded from every group).
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = df[rank_col].notna()
    for _, g in df[valid].groupby(df.loc[valid, image_col]):
        order = g[rank_col].sort_values(kind="mergesort").index
        chunks = np.array_split(np.arange(len(order)), max(1, n_groups))
        for gi, idx in enumerate(chunks, start=1):
            out.loc[order[idx]] = gi
    return out


_DECILE_COLORS = {1: "#d62728", 5: "#1f77b4", 10: "#2ca02c"}
_DECILE_PALETTE_FALLBACK = [
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _decile_color(g: int, highlight: list[int]) -> str:
    if g in _DECILE_COLORS:
        return _DECILE_COLORS[g]
    # deterministic colour for any other highlighted group
    idx = highlight.index(g) % len(_DECILE_PALETTE_FALLBACK)
    return _DECILE_PALETTE_FALLBACK[idx]


def calibration_outputs(
    df: "pd.DataFrame",
    out_plot: Path,
    out_csv: Path | None,
    pred_iou_thresh: float | None = None,
    n_deciles: int = 10,
    highlight_deciles: list[int] | None = None,
) -> None:
    """Full-dataset calibration of SAM's predicted-IoU head: for EVERY case dump
    the (GT IoU, predicted IoU) pair, scatter one vs the other and report the
    correlation between the two full-length vectors.

    Three pairs are reported side by side: the CLEAN (un-attacked, best_box)
    box, the ATTACKED box (bad_box as-is, i.e. what used to be called
    "undefended"), and the final gradient-DEFENDED box.

    Figure layout is 2 rows x 3 cols (one column per clean/attacked/defended):
      row 1 -- coloured by pred_iou_thresh (>= thresh vs < thresh) if given,
               a single colour otherwise; correlation reported overall and
               (if thresholded) per side of the split.
      row 2 -- decile-cluster panel: cases are grouped into n_deciles equal
               groups per image, ranked by ATTACKED IoU (worst -> best, same
               fixed grouping in every column -- see scripts/
               group_robustness_deciles.py); highlight_deciles (default
               G1/G5/G10) are drawn in colour, the rest as a grey background.
    """
    highlight_deciles = highlight_deciles or [1, 5, 10]
    pairs = [
        ("clean (best_box)", "clean_pred", "clean_iou"),
        ("attacked (bad_box)", "undef_pred", "undefended_iou"),
        ("defended (grad_final)", "grad_final_pred", "grad_final_iou"),
    ]

    decile_col = "decile_group"
    df = df.copy()
    df[decile_col] = _assign_deciles(df, "image_name", "undefended_iou", n_deciles)

    if out_csv is not None:
        cols = ["image_name", "kind", "user", "attempt", decile_col,
                "clean_pred", "clean_iou",
                "undef_pred", "undefended_iou",
                "grad_final_pred", "grad_final_iou"]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df[cols].to_csv(out_csv, index=False)
        print(f"Saved gt-vs-pred IoU vectors -> {out_csv}  ({len(df)} cases)")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    print("\n  --- GT IoU vs predicted IoU over ALL cases ---")

    for col, (label, pcol, tcol) in enumerate(pairs):
        sub = df[[pcol, tcol, decile_col]].dropna(subset=[pcol, tcol])
        ax_top, ax_bot = axes[0][col], axes[1][col]
        if not len(sub):
            print(f"  {label:>22}: no finite (pred, gt) pairs, skipping")
            ax_top.axis("off"); ax_bot.axis("off")
            continue

        pred = sub[pcol].to_numpy()
        true = sub[tcol].to_numpy()
        r = _pearson(pred, true)
        rho = _spearman(pred, true)
        print(f"  {label:>22}: n={len(sub)}  Pearson r = {r:.4f}   "
              f"Spearman rho = {rho:.4f}   mean pred {pred.mean():.4f} / "
              f"mean gt {true.mean():.4f}")

        lim = (0.0, max(1.0, float(pred.max(initial=1.0)), float(true.max(initial=1.0))))

        # ---- row 1: threshold two-colour split (or single colour) ----
        if pred_iou_thresh is not None:
            hi = sub[pcol] >= pred_iou_thresh
            lo = ~hi
            for mask, color, tag in ((hi, "#2ca02c", f">=thr"), (lo, "#d62728", f"<thr")):
                if mask.sum() == 0:
                    continue
                p_, t_ = sub.loc[mask, pcol].to_numpy(), sub.loc[mask, tcol].to_numpy()
                r_, rho_ = _pearson(p_, t_), _spearman(p_, t_)
                ax_top.scatter(p_, t_, s=4, alpha=0.2, color=color, edgecolors="none",
                               rasterized=True,
                               label=f"{tag} (n={mask.sum()}, r={r_:.2f}, rho={rho_:.2f})")
                print(f"      {tag:>5} thresh={pred_iou_thresh:.2f}: n={mask.sum()}  "
                      f"r={r_:.4f}  rho={rho_:.4f}")
            ax_top.axvline(pred_iou_thresh, color="k", ls=":", lw=1.0, alpha=0.6)
        else:
            ax_top.scatter(pred, true, s=4, alpha=0.15, color="#1f77b4",
                           edgecolors="none", rasterized=True, label="all cases")

        ax_top.plot(lim, lim, color="#555555", ls="--", lw=1.0, label="y = x")
        ax_top.set_xlim(lim); ax_top.set_ylim(lim)
        ax_top.set_xlabel("SAM predicted IoU"); ax_top.set_ylabel("GT (true) IoU")
        ax_top.set_title(f"{label}   n={len(sub)},  r={r:.3f},  rho={rho:.3f}")
        ax_top.grid(ls=":", alpha=0.4)
        ax_top.legend(loc="upper left", fontsize=7)

        # ---- row 2: decile-cluster panel (G1/G5/G10 by default) ----
        background = sub[~sub[decile_col].isin(highlight_deciles)]
        if len(background):
            ax_bot.scatter(background[pcol], background[tcol], s=4, alpha=0.08,
                           color="#999999", edgecolors="none", rasterized=True,
                           label=f"other deciles (n={len(background)})")
        for g in highlight_deciles:
            grp = sub[sub[decile_col] == g]
            if not len(grp):
                continue
            ax_bot.scatter(grp[pcol], grp[tcol], s=8, alpha=0.5,
                           color=_decile_color(int(g), highlight_deciles),
                           edgecolors="none", rasterized=True,
                           label=f"G{g} (n={len(grp)})")
        ax_bot.plot(lim, lim, color="#555555", ls="--", lw=1.0)
        ax_bot.set_xlim(lim); ax_bot.set_ylim(lim)
        ax_bot.set_xlabel("SAM predicted IoU"); ax_bot.set_ylabel("GT (true) IoU")
        ax_bot.set_title(f"{label} — decile groups (ranked by attacked IoU)")
        ax_bot.grid(ls=":", alpha=0.4)
        ax_bot.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "SAM predicted-IoU calibration: clean vs attacked vs defended boxes\n"
        f"top row: {'threshold split @ ' + str(pred_iou_thresh) if pred_iou_thresh is not None else 'all cases'}"
        f"   |   bottom row: decile clusters {highlight_deciles} of {n_deciles} "
        "(fixed by attacked-box IoU ranking, worst -> best)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration scatter -> {out_plot}")


def _auroc(scores, labels) -> float:
    """AUROC via Mann-Whitney rank statistic. scores: higher => more 'positive'."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum()); n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scores.argsort().argsort().astype(np.float64) + 1  # 1-based ranks
    r_pos = ranks[labels].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def grad_estop_by_stability(traj, predictor, device, use_mm, estop_every, M, sigma_s, seed):
    """(C) walk the gradient trajectory, evaluate stability every N steps, return
    the step that is most stable (early stop without GT). Returns (step, true_iou, stab)."""
    idxs = list(range(0, len(traj), max(1, estop_every)))
    if (len(traj) - 1) not in idxs:
        idxs.append(len(traj) - 1)
    best = None
    for i in idxs:
        t = traj[i]
        stab, _ = stability_score(t["box"], predictor, device, use_mm, t["head"],
                                  M, sigma_s, seed)
        if best is None or stab > best[2]:
            best = (t["step"], t["true_iou"], stab)
    return best


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="critical_shifts",
                   choices=["critical_shifts", "user_study"],
                   help="critical_shifts: best/bad box pairs from JSON; "
                        "user_study: real user boxes from root/{images,masks,user_masks}")
    p.add_argument("--critical_shifts", default="critical_shifts_coco.json")
    p.add_argument("--root", default=None,
                   help="user_study FOR_TEST root (images/, masks/, user_masks/)")
    p.add_argument("--use", default="mp", help="user_study mask kinds: 'm', 'p' or 'mp' (default: both)")
    p.add_argument("--images_dir", default=None,
                   help="critical_shifts: image dir (for user_study derived from --root)")
    p.add_argument("--masks_dir", default=None,
                   help="critical_shifts: GT mask dir (for user_study derived from --root)")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_name", default="SAM",
                   choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"],
                   help="SAM-HQ2/SAM3 run in a separate conda env (see "
                        "scripts/setup_repo.sh + heatmaps/env_dispatch.py); "
                        "this script re-launches itself there automatically. "
                        "SAM3 is not yet wired into the gradient path (see "
                        "heatmaps/comp_hw_smoothed.load_sam3_model).")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=100, help="0 = all cases")

    # gradient refinement
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=3.0, help="Adam lr (~pixels/step)")
    p.add_argument("--multimask", action="store_true", default=False,
                   help="use SAM's 3 multimask heads (pick argmax predicted IoU)")

    # best_of_n comparison
    p.add_argument("--Y", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--sigma_center", type=float, default=0.03)
    p.add_argument("--perturb_mode", default="size", choices=["size", "size_center"])

    # SAIF-style stability (methods A/B/C)
    p.add_argument("--K", type=int, default=5,
                   help="(B) top-K candidates by predicted IoU to re-rank by stability")
    p.add_argument("--stab_M", type=int, default=6,
                   help="number of micro-perturbations to estimate stability")
    p.add_argument("--stab_sigma", type=float, default=0.04,
                   help="relative sigma for stability micro-perturbations")
    p.add_argument("--stab_tau", type=float, default=0.80,
                   help="(A) stability threshold to trust the best_of_n (token-0) result")
    p.add_argument("--estop_every", type=int, default=10,
                   help="(C) evaluate grad-trajectory stability every N steps")
    p.add_argument("--weak_thresh", type=float, default=0.5,
                   help="undefended IoU below this = 'weak' box (detector eval label)")
    p.add_argument("--grad_only", action="store_true", default=False,
                   help="skip the expensive bon_mm / stability-select baselines; "
                        "compute only undefended, best_of_n, gradient + gated rescue")

    p.add_argument("--out_csv", default="results/grad_refine_percase.csv")
    p.add_argument("--out_plot", default="results/grad_refine.png")
    p.add_argument("--steps_csv", default=None,
                   help="optional: per-step aggregate (mean true/pred IoU)")
    p.add_argument("--calib_csv", default="results/gt_vs_pred_iou.csv",
                   help="per-case (GT IoU, predicted IoU) vectors for the "
                        "calibration/correlation analysis ('' to skip)")
    p.add_argument("--calib_plot", default="results/gt_vs_pred_iou.png",
                   help="GT-vs-predicted IoU scatter with Pearson/Spearman "
                        "correlation over all cases")
    p.add_argument("--pred_iou_thresh", type=float, default=None,
                   help="if set, split every calibration scatter into two "
                        "colours by SAM's predicted IoU (>= thresh vs < "
                        "thresh) and report Pearson/Spearman separately for "
                        "each half in addition to the overall correlation")
    p.add_argument("--n_deciles", type=int, default=10,
                   help="number of equal-size groups to split each image's "
                        "cases into, ranked by attacked (undefended) IoU "
                        "(worst -> best), for the decile-cluster panel")
    p.add_argument("--highlight_deciles", type=int, nargs="+", default=[1, 5, 10],
                   help="which decile groups (1..n_deciles) to highlight in "
                        "the decile-cluster panel; the rest are drawn as a "
                        "grey background")
    return p.parse_args()


def load_tasks(args):
    """Return (by_image dict[stem]->list of raw tasks, images_dir, masks_dir, kind).

    critical_shifts: raw task = JSON dict (has bad_box/best_box in 1024 space).
    user_study     : raw task = UserMaskRecord (box computed per-image in-loop).
    """
    if args.dataset == "user_study":
        if not args.root:
            raise SystemExit("--root is required for --dataset user_study")
        root = Path(args.root)
        use = "".join(sorted(set(args.use.lower())))
        recs = index_user_masks(root / "user_masks", kinds=tuple(use))
        if args.limit > 0:
            recs = recs[: args.limit]
        by_image = defaultdict(list)
        for r in recs:
            by_image[r.stem].append(r)
        print(f"Loaded {len(recs)} user masks over {len(by_image)} images from {root}")
        return by_image, str(root / "images"), str(root / "masks"), "user_study"

    if not args.images_dir or not args.masks_dir:
        raise SystemExit("--images_dir and --masks_dir are required for critical_shifts")
    with open(args.critical_shifts) as f:
        shifts = json.load(f)
    if args.limit > 0:
        shifts = shifts[: args.limit]
    by_image = defaultdict(list)
    for c in shifts:
        by_image[c["image_name"]].append(c)
    print(f"Loaded {len(shifts)} critical-shift cases from {args.critical_shifts}")
    return by_image, args.images_dir, args.masks_dir, "critical_shifts"


def build_user_cases(raw_tasks, predictor, gt_tensor):
    """user_study: turn each user mask into a case dict with bad_box (tight user
    bbox, 1024 frame) and best_box (tight GT bbox, 1024 frame = the ideal box)."""
    orig_size = predictor.original_size
    gt_np = gt_tensor.numpy().astype(np.uint8)
    if gt_np.sum() == 0:
        return []
    rmin, rmax, cmin, cmax = get_bbox_from_mask(gt_np)
    gt_bbox = np.array([cmin, rmin, cmax, rmax], dtype=np.int32)
    best_box_1024 = original_to_1024(gt_bbox[None], orig_size)[0]

    cases = []
    for rec in raw_tasks:
        try:
            um = _load_binary_mask(rec.path, target_shape=orig_size)
        except Exception:
            continue
        if um.sum() == 0:
            continue
        rmin, rmax, cmin, cmax = get_bbox_from_mask(um.astype(np.uint8))
        ub = np.array([cmin, rmin, cmax, rmax], dtype=np.int32)
        if ub[2] - ub[0] <= 0 or ub[3] - ub[1] <= 0:
            continue
        bad_box_1024 = original_to_1024(ub[None], orig_size)[0]
        cases.append({"bad_box": bad_box_1024.tolist(),
                      "best_box": best_box_1024.tolist(),
                      "bad_iou": float("nan"), "best_iou": float("nan"),
                      "kind": rec.kind, "user": rec.user, "attempt": rec.attempt})
    return cases


def main():
    args = parse_args()

    # SAM-HQ2 / SAM3 live in separate conda envs (package-name clash with
    # sam2, incompatible torch versions -- see heatmaps/env_dispatch.py).
    # Re-execs into the right env and exits; no-op for the other backends.
    maybe_dispatch_to_env(args.model_name, __file__)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    by_image, images_dir, masks_dir, dataset_kind = load_tasks(args)

    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=device)
    # freeze model params: backward then only flows to the box leaf (cheap)
    for prm in predictor.model.parameters():
        prm.requires_grad_(False)

    sam_h, sam_w = getattr(predictor, "input_size", (1024, 1024))
    rows = []
    true_by_step: dict[int, list[float]] = defaultdict(list)
    pred_by_step: dict[int, list[float]] = defaultdict(list)

    n_done = 0
    for image_name, raw_tasks in by_image.items():
        image_path = _find_file(images_dir, image_name)
        mask_path = _find_file(masks_dir, image_name)
        if image_path is None or mask_path is None:
            print(f"[warn] missing image/mask for {image_name}, skipping")
            continue
        try:
            _prepare_image(str(image_path), predictor)
            gt_tensor = _load_gt(mask_path, predictor)
        except Exception as e:
            print(f"[warn] setup failed for {image_name}: {e}")
            continue

        cases = (build_user_cases(raw_tasks, predictor, gt_tensor)
                 if dataset_kind == "user_study" else raw_tasks)

        for case in cases:
            bad_box = torch.tensor(case["bad_box"], dtype=torch.float32)

            bad_box_np = np.asarray(case["bad_box"], dtype=np.float64)
            best_box_np = np.asarray(case["best_box"], dtype=np.float64)

            # undefended (start point) -- also grab SAM's predicted IoU on bad box
            undef_mask, undef_pred = _predict_single_box(
                bad_box, predictor, device,
                boxes_already_transformed=True, return_score=True,
            )
            undef_iou = _iou(gt_tensor, undef_mask)

            # clean reference: the un-attacked box (best_box), for the
            # clean/attacked/defended calibration comparison
            clean_mask, clean_pred = _predict_single_box(
                torch.tensor(case["best_box"], dtype=torch.float32), predictor, device,
                boxes_already_transformed=True, return_score=True,
            )
            clean_iou = _iou(gt_tensor, clean_mask)

            # best_of_n defence (with scores -> chosen box & its predicted IoU)
            bon_mask, bon_perturbed, bon_scores, bon_best_idx = _predict_smoothed_box(
                bad_box=bad_box, image_shape=(sam_h, sam_w), predictor=predictor,
                rank=device, Y=args.Y, sigma_w=args.sigma, sigma_h=args.sigma,
                averaging_mode="best_of_n", sigma_cx=args.sigma_center,
                sigma_cy=args.sigma_center, perturb_mode=args.perturb_mode, seed=42,
                return_scores=True,
            )
            bon_iou = _iou(gt_tensor, bon_mask)
            bon_pred = float(bon_scores[bon_best_idx, 0]) if bon_best_idx >= 0 else float("nan")
            bon_box = (bon_perturbed[bon_best_idx].numpy() if bon_best_idx >= 0 else bad_box_np)
            bon_m = _box_metrics(bon_box, bad_box_np)

            # best_of_n x multimask: search boxes AND the 3 heads (skipped if grad_only)
            if args.grad_only:
                bon_mm_iou = mm_pred = bon_mm_corner = float("nan"); mm_head = -1
            else:
                mm_mask, mm_pred, mm_box, mm_head = best_of_n_multimask(
                    case["bad_box"], predictor, device, args.Y, args.sigma,
                    args.sigma_center, args.perturb_mode, seed=42)
                bon_mm_iou = _iou(gt_tensor, mm_mask)
                bon_mm_corner = _box_metrics(mm_box, bad_box_np)["corner_l2"]

            # gradient refine from bad_box
            final_box, traj = refine_box_by_iou_grad(
                case["bad_box"], predictor, device,
                steps=args.steps, lr=args.lr, multimask=args.multimask,
                gt_tensor=gt_tensor,
            )
            for t in traj:
                true_by_step[t["step"]].append(t["true_iou"])
                pred_by_step[t["step"]].append(t["pred_score"])
            grad_final_iou = traj[-1]["true_iou"]
            grad_best_iou = max(t["true_iou"] for t in traj)  # best over trajectory
            grad_box = final_box.cpu().numpy()
            grad_m = _box_metrics(grad_box, bad_box_np)

            # step-0 of the trajectory = multimask head-select on bad_box (no grad)
            headsel_iou = traj[0]["true_iou"]
            headsel_pred = traj[0]["pred_score"]

            # (B) stability selector: Y boxes x 4 tokens, top-K by pred, re-rank by stability
            if args.grad_only:
                stab_select_iou = sel_stab = sel_pred = float("nan"); sel_tok = -1
            else:
                sel_mask, sel_stab, sel_box, sel_tok, sel_pred = method_stab_select(
                    case["bad_box"], predictor, device, args.Y, args.sigma,
                    args.sigma_center, args.perturb_mode, seed=42,
                    K=args.K, M=args.stab_M, sigma_s=args.stab_sigma)
                stab_select_iou = _iou(gt_tensor, sel_mask)

            # (A) gate: trust best_of_n (token-0) if it's stable, else multimask rescue
            if args.grad_only:
                gated_iou = stab_bon = oracle_gate_iou = float("nan"); gate_dec = ""
            else:
                stab_bon, _ = stability_score(bon_box, predictor, device, False, 0,
                                              args.stab_M, args.stab_sigma, seed=7)
                if stab_bon >= args.stab_tau:
                    gated_iou, gate_dec = bon_iou, "token0"
                else:
                    gated_iou, gate_dec = bon_mm_iou, "rescue"
                oracle_gate_iou = max(bon_iou, bon_mm_iou)  # gating ceiling (bon vs bon_mm)

            # (C) gradient early-stop by stability
            es_step, es_iou, es_stab = grad_estop_by_stability(
                traj, predictor, device, use_mm=args.multimask,
                estop_every=args.estop_every, M=args.stab_M,
                sigma_s=args.stab_sigma, seed=11)

            # gated gradient-rescue: stability of token-0 on the ORIGINAL box is the
            # gate. Plain: rescue if token-0 unstable. Verified: ALSO require the
            # grad result to be MORE stable than token-0, so false positives revert
            # harmlessly to token-0 (decouples recall from harm).
            stab_undef, _ = stability_score(case["bad_box"], predictor, device,
                                            False, 0, args.stab_M, args.stab_sigma, seed=3)
            rescue_flag = stab_undef < args.stab_tau
            gated_grad_iou = es_iou if rescue_flag else undef_iou
            gg_dec = "grad_rescue" if rescue_flag else "keep"
            accept_v = rescue_flag and (es_stab > stab_undef)
            gated_gradv_iou = es_iou if accept_v else undef_iou
            gg_dec_v = "grad_rescue" if accept_v else "keep"

            rows.append({
                "image_name": image_name,
                # user_study only: which annotation this case came from ('' for critical_shifts)
                "kind": case.get("kind", ""),
                "user": case.get("user", ""),
                "attempt": case.get("attempt", ""),
                "bad_iou_json": float(case["bad_iou"]),
                "best_iou_json": float(case["best_iou"]),
                # clean (un-attacked) reference box, for the clean/attacked/defended
                # calibration comparison (calibration_outputs)
                "clean_iou": clean_iou,
                "clean_pred": clean_pred,
                "undefended_iou": undef_iou,
                "undef_pred": undef_pred,
                "headsel_iou": headsel_iou,
                "headsel_pred": headsel_pred,
                "best_of_n_iou": bon_iou,
                "bon_pred": bon_pred,
                "bon_mm_iou": bon_mm_iou,
                "bon_mm_pred": mm_pred,
                "bon_mm_head": mm_head,
                "bon_mm_corner_l2": bon_mm_corner,
                # (B) stability selector over Y x 4 tokens
                "stab_select_iou": stab_select_iou,
                "stab_select_stab": sel_stab,
                "stab_select_pred": sel_pred,
                "stab_select_tok": sel_tok,
                # (A) stability gate (bon vs bon_mm)
                "gated_iou": gated_iou,
                "gate_decision": gate_dec,
                "stab_bon": stab_bon,
                "oracle_gate_iou": oracle_gate_iou,
                # gated gradient-rescue (keep token-0 if stable, else grad early-stop)
                "stab_undef": stab_undef,
                "gated_grad_iou": gated_grad_iou,
                "gg_decision": gg_dec,
                "gated_gradv_iou": gated_gradv_iou,   # verified (accept grad iff more stable)
                "gg_decision_v": gg_dec_v,
                # (C) gradient early-stop by stability
                "grad_estop_iou": es_iou,
                "grad_estop_step": es_step,
                "grad_estop_stab": es_stab,
                "grad_final_iou": grad_final_iou,
                "grad_best_iou": grad_best_iou,
                "grad_final_pred": traj[-1]["pred_score"],
                "grad_vs_undef": grad_final_iou - undef_iou,
                "grad_vs_bon": grad_final_iou - bon_iou,
                # how far each method moved the box (from bad_box), in 1024 px
                "bon_corner_l2": bon_m["corner_l2"],
                "bon_center_shift": bon_m["center_shift"],
                "bon_w_delta": bon_m["w_delta"],
                "bon_h_delta": bon_m["h_delta"],
                "grad_corner_l2": grad_m["corner_l2"],
                "grad_center_shift": grad_m["center_shift"],
                "grad_w_delta": grad_m["w_delta"],
                "grad_h_delta": grad_m["h_delta"],
                # distance to the reference (good) best_box
                "undef_dist_best": _box_metrics(bad_box_np, best_box_np)["corner_l2"],
                "bon_dist_best": _box_metrics(bon_box, best_box_np)["corner_l2"],
                "grad_dist_best": _box_metrics(grad_box, best_box_np)["corner_l2"],
            })
            n_done += 1

    if n_done == 0:
        raise SystemExit("No cases processed (check paths).")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # full-dataset gt-vs-pred IoU vectors + correlation (Pearson / Spearman)
    calibration_outputs(df, Path(args.calib_plot),
                        Path(args.calib_csv) if args.calib_csv else None,
                        pred_iou_thresh=args.pred_iou_thresh,
                        n_deciles=args.n_deciles,
                        highlight_deciles=args.highlight_deciles)

    steps = sorted(true_by_step)
    true_mean = np.array([np.mean(true_by_step[s]) for s in steps])
    pred_mean = np.array([np.mean(pred_by_step[s]) for s in steps])
    if args.steps_csv:
        sc = Path(args.steps_csv); sc.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"step": steps, "true_iou_mean": true_mean,
                      "pred_iou_mean": pred_mean}).to_csv(sc, index=False)

    _plot(steps, true_mean, pred_mean,
          undef=df["undefended_iou"].mean(), bon=df["best_of_n_iou"].mean(),
          bon_mm=df["bon_mm_iou"].mean(), headsel=df["headsel_iou"].mean(),
          out_path=Path(args.out_plot))

    # ---- summary ----
    print(f"\nProcessed {n_done} cases over {len(by_image)} images "
          f"(multimask={args.multimask}, steps={args.steps}, lr={args.lr})")
    print("=" * 64)
    print(f"  undefended (1 head)      true IoU : {df['undefended_iou'].mean():.4f}"
          f"   (pred {df['undef_pred'].mean():.4f})")
    print(f"  head-select (multimask,1fwd) IoU : {df['headsel_iou'].mean():.4f}"
          f"   (pred {df['headsel_pred'].mean():.4f})   <- NO box change")
    print(f"  best_of_n (1 head)       true IoU : {df['best_of_n_iou'].mean():.4f}"
          f"   (pred {df['bon_pred'].mean():.4f})")
    if not args.grad_only:
        print(f"  best_of_n x multimask    true IoU : {df['bon_mm_iou'].mean():.4f}"
              f"   (pred {df['bon_mm_pred'].mean():.4f})   <- boxes x heads 1-3")
        print(f"  [B] stability-select     true IoU : {df['stab_select_iou'].mean():.4f}"
              f"   (stab {df['stab_select_stab'].mean():.3f})   <- Y x 4 tokens, top-K by stab")
        print(f"  [A] stability-gate       true IoU : {df['gated_iou'].mean():.4f}"
              f"   (token0 {(df['gate_decision'] == 'token0').mean():.0%} / rescue rest)")
        print(f"      oracle-gate (bon|bon_mm) IoU  : {df['oracle_gate_iou'].mean():.4f}  (gating ceiling)")
    print(f"  [C] grad early-stop(stab)true IoU : {df['grad_estop_iou'].mean():.4f}"
          f"   (mean stop step {df['grad_estop_step'].mean():.1f})")
    print(f"  [A+grad] gated grad-rescue   IoU  : {df['gated_grad_iou'].mean():.4f}"
          f"   (rescued {(df['gg_decision'] == 'grad_rescue').mean():.0%})   <- keep token-0 else grad")
    print(f"  [A+grad VERIFIED] g-rescue   IoU  : {df['gated_gradv_iou'].mean():.4f}"
          f"   (rescued {(df['gg_decision_v'] == 'grad_rescue').mean():.0%})   <- accept grad iff more stable")
    print(f"  grad (final)             true IoU : {df['grad_final_iou'].mean():.4f}"
          f"   (pred {df['grad_final_pred'].mean():.4f})")
    print(f"  grad (best/traj)         true IoU : {df['grad_best_iou'].mean():.4f}  (early-stop oracle)")
    print(f"  grad - undef            : {df['grad_vs_undef'].mean():+.4f}")
    print(f"  grad - best_of_n        : {df['grad_vs_bon'].mean():+.4f}")
    print(f"  grad beats best_of_n in : {(df['grad_final_iou'] > df['best_of_n_iou']).mean():.1%} of cases")
    if not args.grad_only:
        print(f"  grad - bon_mm           : {(df['grad_final_iou'] - df['bon_mm_iou']).mean():+.4f}")
        print(f"  grad beats bon_mm    in : {(df['grad_final_iou'] > df['bon_mm_iou']).mean():.1%} of cases")

    # ---- per-10-step trajectory (true & predicted) ----
    print("\n  --- gradient trajectory (every 10th step) ---")
    print(f"  {'step':>5} | {'true IoU':>9} | {'pred IoU':>9}")
    print("  " + "-" * 31)
    tick = list(range(0, args.steps + 1, 10))
    if args.steps not in tick:
        tick.append(args.steps)
    for s in tick:
        if s in true_by_step:
            print(f"  {s:>5} | {np.mean(true_by_step[s]):>9.4f} | {np.mean(pred_by_step[s]):>9.4f}")
    print(f"  {'bon':>5} | {df['best_of_n_iou'].mean():>9.4f} | {df['bon_pred'].mean():>9.4f}"
          "   <- best_of_n reference")

    # ---- where does gradient WIN over best_of_n? ----
    win = df[df["grad_final_iou"] > df["best_of_n_iou"]].copy()
    lose = df[df["grad_final_iou"] <= df["best_of_n_iou"]]
    print(f"\n  --- grad WINS in {len(win)}/{len(df)} cases; "
          f"top by margin (grad - bon): ---")
    print(f"  {'image':>14} {'undef':>6} {'bon':>6} {'grad':>6} {'+marg':>6} "
          f"{'gradMove':>9} {'bonMove':>8} {'grad→best':>9} {'bon→best':>9}")
    for _, r in win.sort_values("grad_vs_bon", ascending=False).head(12).iterrows():
        print(f"  {str(r['image_name'])[:14]:>14} {r['undefended_iou']:>6.3f} "
              f"{r['best_of_n_iou']:>6.3f} {r['grad_final_iou']:>6.3f} "
              f"{r['grad_vs_bon']:>+6.3f} {r['grad_corner_l2']:>9.1f} "
              f"{r['bon_corner_l2']:>8.1f} {r['grad_dist_best']:>9.1f} {r['bon_dist_best']:>9.1f}")

    # ---- box-displacement summary (1024 px) ----
    print("\n  --- box displacement from bad_box (mean, 1024 px) ---")
    print(f"  {'metric':>16} | {'grad':>8} | {'best_of_n':>9}")
    print("  " + "-" * 41)
    for label, gcol, bcol in (
        ("corner L2", "grad_corner_l2", "bon_corner_l2"),
        ("center shift", "grad_center_shift", "bon_center_shift"),
        ("|w_delta|", "grad_w_delta", "bon_w_delta"),
        ("|h_delta|", "grad_h_delta", "bon_h_delta"),
    ):
        gv = df[gcol].abs().mean() if "delta" in gcol else df[gcol].mean()
        bv = df[bcol].abs().mean() if "delta" in bcol else df[bcol].mean()
        print(f"  {label:>16} | {gv:>8.2f} | {bv:>9.2f}")
    print(f"  {'dist→best_box':>16} | {df['grad_dist_best'].mean():>8.2f} "
          f"| {df['bon_dist_best'].mean():>9.2f}   (undef start: {df['undef_dist_best'].mean():.2f})")
    print(f"  movement on WIN cases   grad {win['grad_corner_l2'].mean():.1f}"
          f" vs bon {win['bon_corner_l2'].mean():.1f}"
          f"   | on LOSE cases grad {lose['grad_corner_l2'].mean():.1f}"
          f" vs bon {lose['bon_corner_l2'].mean():.1f}")

    # stratify by initial (undefended) quality -> does it help the WEAK ones?
    q = df["undefended_iou"]
    print("\n  --- stratified by undefended quality ---")
    for name, mask in (("weak  (undef<0.5)", q < 0.5),
                       ("mid   (0.5-0.8) ", (q >= 0.5) & (q < 0.8)),
                       ("strong(undef>=0.8)", q >= 0.8)):
        sub = df[mask]
        if len(sub):
            print(f"  [{name}] n={len(sub):3d}  undef {sub['undefended_iou'].mean():.3f}"
                  f" -> bon {sub['best_of_n_iou'].mean():.3f}"
                  f" -> grad {sub['grad_final_iou'].mean():.3f}"
                  f" -> g-gradV {sub['gated_gradv_iou'].mean():.3f}")

    # stratify by best_of_n quality -> where does each method help / hurt?
    if args.grad_only:
        methods = [("undef", "undefended_iou"), ("bon", "best_of_n_iou"),
                   ("gradF", "grad_final_iou"), ("gradES", "grad_estop_iou"),
                   ("g-grad", "gated_grad_iou"), ("g-gradV", "gated_gradv_iou")]
    else:
        methods = [("bon", "best_of_n_iou"), ("bon_mm", "bon_mm_iou"),
                   ("B:stab", "stab_select_iou"), ("A:gate", "gated_iou"),
                   ("C:gES", "grad_estop_iou"), ("g-gradV", "gated_gradv_iou"),
                   ("oracle", "oracle_gate_iou")]
    b = df["best_of_n_iou"]
    print("\n  --- stratified by best_of_n quality (per-method true IoU) ---")
    hdr = f"  {'bucket':>16} | {'n':>3} | " + " | ".join(f"{lbl:>7}" for lbl, _ in methods)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, mask in (("bon<0.2 (dead) ", b < 0.2),
                       ("bon 0.2-0.5    ", (b >= 0.2) & (b < 0.5)),
                       ("bon 0.5-0.8    ", (b >= 0.5) & (b < 0.8)),
                       ("bon>=0.8 (good)", b >= 0.8)):
        sub = df[mask]
        if len(sub):
            vals = " | ".join(f"{sub[col].mean():>7.3f}" for _, col in methods)
            print(f"  {name:>16} | {len(sub):>3} | {vals}")

    # ---- weakness-detector eval: can token-0 stability flag boxes needing rescue? ----
    weak = (df["undefended_iou"] < args.weak_thresh).to_numpy()
    # low stability => weak, so the detector score is (1 - stab_undef)
    auc = _auroc(1.0 - df["stab_undef"].to_numpy(), weak)
    print(f"\n  --- weakness detector: token-0 stability vs (undef < {args.weak_thresh}) ---")
    print(f"  weak boxes: {weak.sum()}/{len(df)}   AUROC(low stab -> weak) = {auc:.3f}")
    print(f"  {'tau':>5} | {'rescued%':>8} | {'recall':>6} | {'prec':>6} | "
          f"{'gated IoU':>9} | {'verified':>9}")
    print("  " + "-" * 58)
    undef_arr = df["undefended_iou"].to_numpy()
    es_arr = df["grad_estop_iou"].to_numpy()
    stab_arr = df["stab_undef"].to_numpy()
    esstab_arr = df["grad_estop_stab"].to_numpy()
    for tau in (0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99):
        rescue = stab_arr < tau
        gated = np.where(rescue, es_arr, undef_arr)
        # verified: only accept the grad result if it is more stable than token-0
        gated_v = np.where(rescue & (esstab_arr > stab_arr), es_arr, undef_arr)
        tp = int((rescue & weak).sum())
        recall = tp / max(1, int(weak.sum()))
        prec = tp / max(1, int(rescue.sum()))
        print(f"  {tau:>5.2f} | {rescue.mean():>7.1%} | {recall:>6.2f} | {prec:>6.2f} | "
              f"{gated.mean():>9.4f} | {gated_v.mean():>9.4f}")
    print(f"  {'(undef baseline)':>22} mean IoU = {undef_arr.mean():.4f}")
    print(f"\nSaved per-case -> {out_csv}")


def _plot(steps, true_mean, pred_mean, undef, bon, bon_mm, headsel, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(steps, true_mean, "-o", color="#1f77b4", label="grad-refine (true IoU)")
    ax.axhline(undef, color="#d62728", ls="--", label=f"undefended ({undef:.3f})")
    ax.axhline(headsel, color="#ff7f0e", ls=":", label=f"head-select 1fwd ({headsel:.3f})")
    ax.axhline(bon, color="#2ca02c", ls="--", label=f"best_of_n ({bon:.3f})")
    ax.axhline(bon_mm, color="#8c564b", ls="-.", label=f"best_of_n x mm ({bon_mm:.3f})")
    ax.set_xlabel("gradient step"); ax.set_ylabel("mean true IoU")
    ax.set_title("True IoU vs gradient step"); ax.grid(ls=":", alpha=0.4); ax.legend()

    ax = axes[1]
    ax.plot(steps, pred_mean, "-o", color="#9467bd", label="predicted IoU (head)")
    ax.plot(steps, true_mean, "-o", color="#1f77b4", label="true IoU")
    ax.set_xlabel("gradient step"); ax.set_ylabel("mean IoU")
    ax.set_title("Predicted vs true IoU (calibration check)")
    ax.grid(ls=":", alpha=0.4); ax.legend()

    fig.suptitle("Gradient ascent on SAM predicted-IoU: does true IoU follow?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
