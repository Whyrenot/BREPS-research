"""
visualize_attention_maps.py
============================
Task 4 of the day's plan: decoder cross-attention maps for a CLEAN
(best_box) vs an ATTACKED (bad_box) prompt on the same image -- where in the
image does each mask-decoder layer's query token look, and how does that
shift once the box prompt is adversarially perturbed?

SAM's mask decoder is a TwoWayTransformer: at every layer the token stack
(iou token + mask tokens + prompt tokens) cross-attends to the flattened
image embedding via a `cross_attn_token_to_image` Attention module (plus one
more `final_attn_token_to_image` after the last layer). That attention
matrix IS the interpretable "where does this token look" map -- but
`Attention.forward` (segment_anything/modeling/transformer.py, reused
near-verbatim by SAM-HQ and SAM2.1) only returns the attention OUTPUT, not
the softmax weights themselves. So this script monkeypatches `.forward` on
just those Attention instances (recomputing the exact same q/k/v projection
+ softmax the module already does) to also stash the weight matrix -- no
change to the model's actual output, just an extra read of an intermediate
tensor. See `_hook_cross_attn`.

For one query token (`--token_idx`, default 1 = the single-output mask
token used when multimask_output=False) we get, per layer, an attention
distribution over the H_e x W_e image-embedding grid (64x64 for a
1024-input SAM). Upsampled to image resolution and overlaid (jet colormap)
on the photo: CLEAN | ATTACKED | signed diff, one row per hooked layer.

Also dumps a per-layer scalar: total-variation distance between the clean
and attacked attention distributions (computed on the raw, un-upsampled
probability vectors, so it's an honest distance between two softmax
outputs), averaged over all cases -- a quantitative complement to the maps
answering "which layer's attention is most disrupted by the attack".

Example (~30 cases, one PNG per case + an aggregate summary):
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_attention_maps.py \
        --critical_shifts critical_shifts_coco.json \
        --images_dir /.../COCO_MVal/img \
        --checkpoint_path /.../sam_vit_b_01ec64.pth \
        --model_name SAM --model_type vit_b \
        --limit 30 \
        --out_dir visualizations/attention_maps \
        --summary_csv results/attention_shift_by_layer.csv \
        --summary_plot visualizations/attention_shift_by_layer.png
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
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

from heatmaps.comp_hw_smoothed import get_bbox_from_mask, get_original_size, load_model
from heatmaps.defend_critical_shifts import (
    _find_file,
    _predict_single_box,
    _prepare_image,
    boxes_to_original,
    original_to_1024,
)
from heatmaps.defend_user_study import _load_binary_mask, index_user_masks
from heatmaps.env_dispatch import maybe_dispatch_to_env
from heatmaps.predicted_iou_heatmap import _blend, _load_display_image

_TOKEN_NAMES = {0: "iou_token", 1: "mask_token0 (single-output)",
                2: "mask_token1", 3: "mask_token2", 4: "mask_token3"}


# ---------------------------------------------------------------------------
# dataset loading: critical_shifts JSON (bad_box/best_box already in 1024
# space) or user_study (root/{images,masks,user_masks}, boxes derived here --
# mirrors scripts/refine_box_iou_grad.py's load_tasks()/build_user_cases()).
# ---------------------------------------------------------------------------

def _load_gt(mask_path, predictor) -> torch.Tensor:
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(mask_path)
    gt = (gt > 0).astype(np.uint8)
    H, W = get_original_size(predictor)
    if gt.shape != (H, W):
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(gt > 0)


def _build_user_cases(raw_tasks, predictor, gt_tensor) -> "list[dict]":
    """user_study: each user annotation -> a case dict with bad_box (tight
    user bbox, "attacked") and best_box (tight GT bbox, "clean"), both in
    SAM's 1024-input frame -- same convention as the critical_shifts JSON."""
    orig_size = get_original_size(predictor)
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
        cases.append({"best_box": best_box_1024.tolist(), "bad_box": bad_box_1024.tolist(),
                      "kind": rec.kind, "user": rec.user, "attempt": rec.attempt})
    return cases


# ---------------------------------------------------------------------------
# attention capture: monkeypatch just the token->image cross-attn modules
# ---------------------------------------------------------------------------

_CAPTURED_ATTN: dict[str, torch.Tensor] = {}
_CAPTURED_SHAPE: dict[str, tuple[int, int]] = {}


def _find_cross_attn_modules(transformer) -> "list[tuple[str, torch.nn.Module]]":
    """Every `cross_attn_token_to_image` (per layer) + `final_attn_token_to_image`
    Attention submodule, in forward execution order. Filtered by NAME (not
    class identity) so this works across SAM / SAM-HQ / SAM2.1 forks that
    reuse the same transformer.py structure under possibly different
    import paths."""
    wanted = []
    for name, mod in transformer.named_modules():
        if name.endswith("cross_attn_token_to_image") or name == "final_attn_token_to_image":
            wanted.append((name, mod))
    return wanted


def _make_capturing_forward(module: torch.nn.Module, name: str):
    """Reimplements Attention.forward's math (q/k/v proj -> heads -> softmax
    -> out_proj) so we can additionally stash the softmax weights; returns
    the exact same output tensor the original forward would."""

    def _forward(q, k, v):
        qp = module.q_proj(q)
        kp = module.k_proj(k)
        vp = module.v_proj(v)
        qh = module._separate_heads(qp, module.num_heads)
        kh = module._separate_heads(kp, module.num_heads)
        vh = module._separate_heads(vp, module.num_heads)
        c_per_head = qh.shape[-1]
        attn = (qh @ kh.transpose(-2, -1)) / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        _CAPTURED_ATTN[name] = attn.detach()  # (B, num_heads, N_tokens, N_image)
        out = attn @ vh
        out = module._recombine_heads(out)
        out = module.out_proj(out)
        return out

    return _forward


def _transformer_shape_prehook(mod, args):
    """TwoWayTransformer.forward(image_embedding, image_pe, point_embedding),
    all positional -- args[0] is the (B, C, H, W) image embedding, before the
    flatten/permute done inside forward()."""
    if args:
        _CAPTURED_SHAPE["hw"] = tuple(int(s) for s in args[0].shape[-2:])


@contextlib.contextmanager
def _hook_cross_attn(model):
    """Patch every token->image cross-attn Attention instance in
    model.mask_decoder.transformer to also record its softmax weights.
    Yields the list of hooked layer names (execution order). Modules missing
    the expected sub-attributes (unexpected backend internals) are skipped
    with a warning rather than crashing."""
    transformer = model.mask_decoder.transformer
    targets = _find_cross_attn_modules(transformer)

    originals: dict[str, object] = {}
    hooked_names: list[str] = []
    required = ("q_proj", "k_proj", "v_proj", "out_proj", "num_heads",
                "_separate_heads", "_recombine_heads")
    for name, mod in targets:
        if not all(hasattr(mod, a) for a in required):
            print(f"[visualize_attention_maps] WARNING: '{name}' is missing "
                  f"expected Attention internals, skipping capture for it.",
                  file=sys.stderr)
            continue
        originals[name] = mod.__dict__.get("forward")  # None if not instance-patched yet
        mod.forward = _make_capturing_forward(mod, name)
        hooked_names.append(name)

    shape_handle = transformer.register_forward_pre_hook(_transformer_shape_prehook)
    try:
        yield hooked_names
    finally:
        shape_handle.remove()
        for name, mod in targets:
            if name not in originals:
                continue
            if originals[name] is None:
                del mod.__dict__["forward"]  # fall back to the class method
            else:
                mod.forward = originals[name]


def capture_attention_for_box(box_1024, predictor, device) -> "tuple[dict, tuple[int, int]]":
    """Run a single box through the model with cross-attn capture active.
    Returns ({layer_name: attn (num_heads, N_tokens, N_image) cpu float32},
    (H_e, W_e))."""
    _CAPTURED_ATTN.clear()
    _CAPTURED_SHAPE.clear()
    model = getattr(predictor, "model", predictor)
    with _hook_cross_attn(model):
        box_t = torch.as_tensor(box_1024, dtype=torch.float32)
        _predict_single_box(box_t, predictor, device, boxes_already_transformed=True)
    snap = {k: v[0].float().cpu() for k, v in _CAPTURED_ATTN.items()}  # drop batch dim
    hw = _CAPTURED_SHAPE.get("hw")
    if hw is None:
        # square-grid fallback if the pre-hook didn't fire for some backend
        n_image = next(iter(snap.values())).shape[-1] if snap else 0
        side = int(round(math.sqrt(n_image)))
        hw = (side, side)
    return snap, hw


# ---------------------------------------------------------------------------
# per-case figure
# ---------------------------------------------------------------------------

def _token_map(attn_layer: torch.Tensor, token_idx: int, hw: tuple[int, int]) -> np.ndarray:
    """attn_layer: (num_heads, N_tokens, N_image). Mean over heads, row for
    `token_idx`, reshaped to (H_e, W_e). Returns a float32 probability map
    (sums to ~1 over H_e*W_e)."""
    row = attn_layer.mean(dim=0)[token_idx]  # (N_image,)
    H_e, W_e = hw
    return row.reshape(H_e, W_e).numpy().astype(np.float32)


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two discrete distributions (both
    already sum to ~1): 0.5 * sum(|p - q|), in [0, 1]."""
    return float(0.5 * np.abs(p.reshape(-1) - q.reshape(-1)).sum())


def _upsample(m: np.ndarray, H: int, W: int) -> np.ndarray:
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)


def _overlay(disp: np.ndarray, norm_map: np.ndarray, cmap_name: str,
             vmin: float, vmax: float, alpha: float) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    scaled = np.clip((norm_map - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgb = (cmap(scaled)[..., :3] * 255).astype(np.uint8)
    cover = np.ones(disp.shape[:2], dtype=bool)
    return _blend(disp, rgb, cover, alpha)


def _draw_box(img: np.ndarray, box_orig, color) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = (int(round(v)) for v in box_orig)
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out


def make_case_figure(image_path, best_box_1024, bad_box_1024, predictor, device,
                     token_idx: int, alpha: float, out_path: Path,
                     case_label: str) -> "dict[str, float]":
    """One PNG: rows = hooked decoder layers, cols = [clean, attacked, diff].
    Returns {layer_name: tv_distance} for the summary aggregation."""
    H, W = get_original_size(predictor)
    disp = _load_display_image(image_path, H, W)

    best_orig = boxes_to_original(np.asarray(best_box_1024)[None], (H, W))[0]
    bad_orig = boxes_to_original(np.asarray(bad_box_1024)[None], (H, W))[0]

    attn_clean, hw = capture_attention_for_box(best_box_1024, predictor, device)
    attn_attacked, hw2 = capture_attention_for_box(bad_box_1024, predictor, device)
    if hw != hw2:
        raise RuntimeError(f"image-embedding grid changed between boxes ({hw} vs {hw2}) "
                            f"for {case_label} -- unexpected, skipping")

    layers = [l for l in attn_clean.keys() if l in attn_attacked]
    if not layers:
        raise RuntimeError(f"no common hooked layers captured for {case_label}")

    n_tokens = attn_clean[layers[0]].shape[1]
    if token_idx >= n_tokens:
        raise RuntimeError(f"--token_idx {token_idx} out of range (only {n_tokens} tokens)")

    tv_by_layer: dict[str, float] = {}
    fig, axes = plt.subplots(len(layers), 3, figsize=(13, 4.2 * len(layers)),
                             squeeze=False, constrained_layout=True)

    for i, layer in enumerate(layers):
        m_clean = _token_map(attn_clean[layer], token_idx, hw)
        m_attacked = _token_map(attn_attacked[layer], token_idx, hw)
        tv_by_layer[layer] = _tv_distance(m_clean, m_attacked)

        up_clean = _upsample(m_clean, H, W)
        up_attacked = _upsample(m_attacked, H, W)
        vmax = max(up_clean.max(), up_attacked.max())

        clean_img = _draw_box(_overlay(disp, up_clean, "jet", 0.0, vmax, alpha),
                              best_orig, (0, 255, 0))
        attacked_img = _draw_box(_overlay(disp, up_attacked, "jet", 0.0, vmax, alpha),
                                 bad_orig, (255, 0, 0))

        diff = up_attacked - up_clean
        dlim = max(abs(diff.min()), abs(diff.max()), 1e-9)
        diff_img = _overlay(disp, diff, "RdBu_r", -dlim, dlim, alpha)

        axes[i][0].imshow(clean_img); axes[i][0].axis("off")
        axes[i][0].set_title(f"{layer}\nCLEAN (best_box, green)", fontsize=9)
        axes[i][1].imshow(attacked_img); axes[i][1].axis("off")
        axes[i][1].set_title("ATTACKED (bad_box, red)", fontsize=9)
        axes[i][2].imshow(diff_img); axes[i][2].axis("off")
        axes[i][2].set_title(f"diff (attacked - clean)   TV={tv_by_layer[layer]:.3f}",
                             fontsize=9)

    fig.suptitle(f"Decoder cross-attention (token->image), token="
                f"'{_TOKEN_NAMES.get(token_idx, token_idx)}' -- {case_label}",
                fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tv_by_layer


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="critical_shifts",
                   choices=["critical_shifts", "user_study"],
                   help="critical_shifts: best/bad box pairs from JSON; "
                        "user_study: real user boxes from --root/{images,masks,user_masks}")
    p.add_argument("--critical_shifts", default="critical_shifts_coco.json")
    p.add_argument("--root", default=None,
                   help="user_study root (images/, masks/, user_masks/); required "
                        "for --dataset user_study")
    p.add_argument("--use", default="mp", help="user_study mask kinds: 'm', 'p' or 'mp'")
    p.add_argument("--images_dir", default=None,
                   help="critical_shifts: image dir (for user_study derived from --root)")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_name", default="SAM",
                   choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"],
                   help="SAM-HQ2/SAM3 auto re-launch in their own conda env, "
                        "see heatmaps/env_dispatch.py")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=30,
                   help="number of critical-shift cases to render (default 30, "
                        "matching the day's plan: ~30 images)")
    p.add_argument("--token_idx", type=int, default=1,
                   help="0=iou token, 1=single-output mask token (default), "
                        "2..N-1=multimask heads")
    p.add_argument("--alpha", type=float, default=0.5, help="heatmap overlay opacity")
    p.add_argument("--out_dir", default="visualizations/attention_maps")
    p.add_argument("--summary_csv", default="results/attention_shift_by_layer.csv")
    p.add_argument("--summary_plot", default="visualizations/attention_shift_by_layer.png")
    return p.parse_args()


def main():
    args = parse_args()
    maybe_dispatch_to_env(args.model_name, __file__)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- load cases, grouped by image (so each image is encoded once) ----
    masks_dir = None
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
        images_dir, masks_dir = str(root / "images"), str(root / "masks")
        print(f"Rendering attention maps for {len(recs)} user-study annotations "
              f"over {len(by_image)} images from {root}")
    else:
        if not args.images_dir:
            raise SystemExit("--images_dir is required for --dataset critical_shifts")
        with open(args.critical_shifts, "r") as f:
            shifts = json.load(f)
        if args.limit > 0:
            shifts = shifts[: args.limit]
        by_image = defaultdict(list)
        for c in shifts:
            by_image[c["image_name"]].append(c)
        images_dir = args.images_dir
        print(f"Rendering attention maps for {len(shifts)} critical-shift cases "
              f"from {args.critical_shifts}")

    total_cases = sum(len(v) for v in by_image.values())
    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=device)

    out_dir = Path(args.out_dir)
    tv_acc: dict[str, list[float]] = defaultdict(list)
    layer_order: list[str] = []
    n_done = 0

    for image_name, raw_tasks in by_image.items():
        image_path = _find_file(images_dir, image_name)
        if image_path is None:
            print(f"[warn] image not found: {image_name}, skipping")
            continue
        try:
            _prepare_image(str(image_path), predictor)
        except Exception as e:
            print(f"[warn] encode failed for {image_name}: {e}")
            continue

        if args.dataset == "user_study":
            mask_path = _find_file(masks_dir, image_name)
            if mask_path is None:
                print(f"[warn] no GT mask for {image_name}, skipping")
                continue
            gt_tensor = _load_gt(mask_path, predictor)
            cases = _build_user_cases(raw_tasks, predictor, gt_tensor)
        else:
            cases = raw_tasks

        stem = Path(image_name).stem
        for j, case in enumerate(cases):
            tag = "_".join(str(case[k]) for k in ("kind", "user", "attempt") if case.get(k) != "") \
                if args.dataset == "user_study" else f"{j:03d}"
            case_label = f"{stem}_{tag}" if tag else f"{stem}_{n_done:03d}"
            out_path = out_dir / f"attn_{case_label}.png"
            try:
                tv_by_layer = make_case_figure(
                    image_path, case["best_box"], case["bad_box"], predictor, device,
                    token_idx=args.token_idx, alpha=args.alpha,
                    out_path=out_path, case_label=case_label,
                )
            except Exception as e:
                print(f"[warn] failed on {case_label}: {e}")
                continue

            for layer, tv in tv_by_layer.items():
                if layer not in layer_order:
                    layer_order.append(layer)
                tv_acc[layer].append(tv)
            n_done += 1
            print(f"  [{n_done}/{total_cases}] {case_label} -> {out_path}")

    if n_done == 0:
        raise SystemExit("No cases rendered (check --images_dir/--root paths).")

    # ---- aggregate summary ----
    rows = [{"order": i, "layer": layer, "n": len(tv_acc[layer]),
             "tv_mean": float(np.mean(tv_acc[layer])), "tv_std": float(np.std(tv_acc[layer]))}
            for i, layer in enumerate(layer_order)]
    df = pd.DataFrame(rows)
    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    print(f"\nSaved per-layer attention-shift summary -> {summary_csv}")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(layer_order)), 5))
    x = np.arange(len(layer_order))
    ax.errorbar(x, df["tv_mean"], yerr=df["tv_std"], fmt="-o", color="#d62728",
               capsize=4, lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(layer_order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("TV distance (clean vs attacked attention)")
    ax.set_title(f"Cross-attention shift under attack, by decoder layer "
                f"(token='{_TOKEN_NAMES.get(args.token_idx, args.token_idx)}', "
                f"n={n_done} cases)")
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    summary_plot = Path(args.summary_plot)
    summary_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary_plot, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot -> {summary_plot}")
    print(f"\nRendered {n_done}/{total_cases} case figures -> {out_dir}/")


if __name__ == "__main__":
    main()
