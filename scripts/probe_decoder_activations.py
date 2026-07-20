"""
probe_decoder_activations.py
============================
Where does a *critical shift* explode inside SAM?

A critical-shift case is a pair of spatially adjacent box prompts
(best_box, bad_box) (<=10 px apart) for the SAME image, where best_box gives
a good mask and bad_box a broken one (see scripts/find_critical_shifts.py).
Because the image is identical within a pair, the image embedding is identical
too -- ALL divergence is born in the prompt encoder and amplified through the
mask-decoder layers.

For each of the first N pairs we:
    1. encode the image once,
    2. run best_box, capture activations of EVERY submodule of the
       prompt_encoder and mask_decoder (forward hooks),
    3. run bad_box (and optionally a CONTROL box), capture the same,
    4. per layer compute, on the activation DIFFERENCE of the pair:
           raw_l2 = || A_best - A_other ||_2
       plus three normalised variants for cross-layer comparability:
           rel_l2 = raw_l2 / (|| A_best ||_2 + eps)        (relative L2)
           rms_l2 = raw_l2 / sqrt(numel)                    (per-element RMS)
           l1_rel = || A_best - A_other ||_1 / (|| A_best ||_1 + eps)  (relative L1)
    5. per layer also record the std of the activation VALUES themselves for
       each box type (normal=best_box, critical=bad_box, control=control_box),
    6. average each metric over all pairs.

Outputs:
    * --out_csv      : per-(layer, pair_type) L2/L1 divergence table.
    * --out_plot     : raw + relative L2 vs layer; each curve is the mean over
                       pairs with a running mean±std band drawn around it.
    * --out_l1_plot  : relative L1 vs layer, same mean±std band.
    * --out_std_csv  : per-layer std of activation values, one column per
                       scenario (normal / critical / control) -- a wide table.

LAYER NAMING (matches the authors' code, facebookresearch/segment-anything)
------------
Only modules whose outputs are actually USED in a single-box forward pass
(multimask_output=False) are hooked, strictly in forward execution order:

    prompt_encoder[sparse_embeddings]  box-corner embeddings (BOX-DEPENDENT)
    prompt_encoder[dense_embeddings]   no_mask_embed         (BOX-INDEPENDENT)
    mask_decoder.<leaf module>         every used leaf of the decoder, e.g.
                                       transformer.layers.0.self_attn.q_proj
    mask_decoder[masks]                predicted mask logits (final output)
    mask_decoder[iou_pred]             predicted-IoU score   (final output)

Leaf names come from named_modules(), so they match the authors' attribute
names verbatim; the bracketed suffixes are the exact variable names returned
by the authors' forward():  PromptEncoder -> (sparse_embeddings,
dense_embeddings),  MaskDecoder -> (masks, iou_pred).

Excluded as UNUSED (this answers "visualize only the used layers"):
    * container modules -- their hook output is the same tensor as their last
      child's, i.e. pure duplicates (self_attn == self_attn.out_proj, etc.);
    * prompt_encoder.pe_layer -- its forward fires only via get_dense_pe()
      (positional encoding of the image grid), box-independent;
    * output_hypernetworks_mlps.1..3 -- they run, but their outputs are
      discarded when multimask_output=False (mask_slice = slice(0, 1) in the
      authors' MaskDecoder.forward).

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/probe_decoder_activations.py \
        --critical_shifts critical_shifts.json \
        --images_dir /.../FOR_TEST/images \
        --checkpoint_path /.../sam_vit_b_01ec64.pth \
        --model_name SAM --model_type vit_b \
        --limit 100 --control \
        --out_csv outputs_smoothed/decoder_activation_l2.csv \
        --out_plot visualizations/decoder_activation_l2.png \
        --out_l1_plot visualizations/decoder_activation_l1.png \
        --out_std_csv outputs_smoothed/decoder_activation_std.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heatmaps.comp_hw_smoothed import load_model
from heatmaps.defend_critical_shifts import _find_file, _predict_single_box, _prepare_image


# ---------------------------------------------------------------------------
# canonical layer naming
# ---------------------------------------------------------------------------

def _semantic_out_names(full: str) -> list[str] | None:
    """Names for the elements of a hooked root module's TUPLE output -- exactly
    the variable names returned by the authors' forward()
    (segment_anything/modeling/{prompt_encoder,mask_decoder}.py)."""
    if full == "prompt_encoder":
        return ["sparse_embeddings", "dense_embeddings"]
    if full == "mask_decoder":
        return ["masks", "iou_pred"]
    return None


# Modules that RUN during a single-box predict but whose outputs are NOT used:
#   * pe_layer: its forward fires only via prompt_encoder.get_dense_pe() (the
#     image-grid positional encoding passed to the decoder as image_pe); the
#     box path uses pe_layer.forward_with_coords(), which a hook does not see.
#     Box-independent -> divergence is 0 by construction.
#   * output_hypernetworks_mlps.1..3: with multimask_output=False the authors'
#     MaskDecoder.forward keeps mask_slice = slice(0, 1), so only mlp 0
#     contributes to the returned mask.
_UNUSED_RE = re.compile(
    r"(?:^|\.)pe_layer(?:$|\.)|(?:^|\.)output_hypernetworks_mlps\.[123](?:$|\.)"
)


def _group_of(layer: str) -> str:
    """Top-level group of a (possibly renamed) layer key."""
    return "prompt_encoder" if layer.startswith("prompt_encoder") else "mask_decoder"


# ---------------------------------------------------------------------------
# activation capture
# ---------------------------------------------------------------------------

# filled by the forward hooks during a single predict call, then snapshotted
_CAPTURED: "OrderedDict[str, torch.Tensor]" = OrderedDict()


def _store(name: str, out, out_names: list[str] | None = None) -> None:
    """Store tensor output(s) of a module. Tuples/lists -> per-element keys.

    out_names: optional list of the authors' return-value names for tuple
        elements (e.g. ["sparse_embeddings", "dense_embeddings"]); produces
        keys like `prompt_encoder[sparse_embeddings]`. Falls back to generic
        `.out0`/`.out1` when no semantic name is available."""
    if torch.is_tensor(out):
        _CAPTURED[name] = out.detach().float().reshape(-1).cpu()
    elif isinstance(out, (tuple, list)):
        for i, o in enumerate(out):
            if torch.is_tensor(o):
                if out_names is not None and i < len(out_names):
                    key = f"{name}[{out_names[i]}]"
                else:
                    key = f"{name}.out{i}"
                _CAPTURED[key] = o.detach().float().reshape(-1).cpu()


def register_hooks(model) -> tuple[list, list[str]]:
    """Hook exactly the modules whose outputs are USED in a single-box forward
    pass (multimask_output=False); captured in forward execution order.

    Hooked:
      * prompt_encoder root -> (sparse_embeddings, dense_embeddings). For a box
        prompt no child of the prompt_encoder fires a forward (the box enters
        via forward_with_coords / embedding .weight), so the root output is the
        only prompt-level signal.
      * every used LEAF module of mask_decoder. Containers are skipped: their
        output is the same tensor as their last child's (pure duplicates).
      * mask_decoder root -> (masks, iou_pred), the final model outputs.

    Skipped as unused: see _UNUSED_RE.
    """
    prompt_encoder = getattr(model, "prompt_encoder", None)
    mask_decoder = getattr(model, "mask_decoder", None)
    if prompt_encoder is None or mask_decoder is None:
        raise SystemExit(
            "Could not find .prompt_encoder / .mask_decoder on the model. "
            f"Available attrs: {[a for a in dir(model) if not a.startswith('_')][:30]}"
        )

    hooks = []
    groups = []
    for group, root in (("prompt_encoder", prompt_encoder), ("mask_decoder", mask_decoder)):
        for name, mod in root.named_modules():
            is_root = (name == "")
            is_leaf = next(mod.children(), None) is None
            if not is_root and (not is_leaf or _UNUSED_RE.search(name)):
                continue
            full = f"{group}.{name}" if name else group
            out_names = _semantic_out_names(full)
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, _n=full, _on=out_names: _store(_n, o, _on)
            ))
            groups.append(full)
    return hooks, groups


def capture_for_box(box_1024, predictor, device) -> "OrderedDict[str, torch.Tensor]":
    """Run a single box through the model; return a snapshot of all activations."""
    _CAPTURED.clear()
    box_t = torch.tensor(box_1024, dtype=torch.float32)
    _predict_single_box(box_t, predictor, device, boxes_already_transformed=True)
    return OrderedDict(_CAPTURED)  # shallow copy of the per-call snapshot


def make_control_box(best_box, bad_box, rng):
    """A *control* neighbour of best_box: a random shift of the SAME magnitude as
    the critical (best->bad) displacement, but in a generic direction. Lets us
    compare 'same-size box perturbation that does NOT break the model' against
    the adversarial one. Returns (control_box, displacement_budget)."""
    best = np.asarray(best_box, dtype=np.float64)
    bad = np.asarray(bad_box, dtype=np.float64)
    d = int(max(1, np.abs(best - bad).max()))  # matched perturbation budget (px)
    for _ in range(20):
        ctrl = best + rng.integers(-d, d + 1, size=4)
        if ctrl[2] - ctrl[0] > 1 and ctrl[3] - ctrl[1] > 1:  # valid box
            return ctrl.tolist(), d
    return best.tolist(), d  # fallback: no shift


def divergence(A_ref, A_other, eps=1e-12) -> dict:
    """Per-layer divergence of the activation difference (A_ref vs A_other):
    raw L2, relative L2, per-element RMS, relative L1."""
    out = {}
    for layer, a in A_ref.items():
        b = A_other.get(layer)
        if b is None or b.shape != a.shape:
            continue
        diff = a - b
        raw = float(torch.norm(diff, p=2).item())
        rel = raw / (float(torch.norm(a, p=2).item()) + eps)
        rms = raw / math.sqrt(a.numel())
        l1_rel = float(torch.norm(diff, p=1).item()) / (float(torch.norm(a, p=1).item()) + eps)
        out[layer] = (raw, rel, rms, l1_rel)
    return out


def activation_std(A) -> dict:
    """Per-layer std of the activation VALUES themselves (spread of A)."""
    return {layer: float(a.std().item()) for layer, a in A.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--critical_shifts", default="critical_shifts.json")
    p.add_argument("--images_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_name", default="SAM",
                   choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=100,
                   help="number of critical-shift pairs to probe (0 = all)")
    p.add_argument("--control", action="store_true", default=False,
                   help="also probe a CONTROL neighbour (random shift of the same "
                        "magnitude as best->bad) and overlay it, to show divergence "
                        "is specific to the adversarial direction")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the control-box random shifts")
    p.add_argument("--out_csv", default="outputs_smoothed/decoder_activation_l2.csv")
    p.add_argument("--out_plot", default="visualizations/decoder_activation_l2.png")
    p.add_argument("--out_l1_plot", default="visualizations/decoder_activation_l1.png")
    p.add_argument("--out_std_csv", default="outputs_smoothed/decoder_activation_std.csv")
    p.add_argument("--per_case_csv", default=None,
                   help="optional: also dump per-(pair,layer) raw L2 / L1")
    return p.parse_args()


def main():
    args = parse_args()
    from heatmaps.env_dispatch import maybe_dispatch_to_env
    maybe_dispatch_to_env(args.model_name, __file__)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(args.critical_shifts, "r") as f:
        shifts = json.load(f)
    if args.limit > 0:
        shifts = shifts[: args.limit]
    print(f"Probing {len(shifts)} critical-shift pairs from {args.critical_shifts}")

    predictor = load_model(
        model_name=args.model_name, model_type=args.model_type,
        checkpoint=args.checkpoint_path, device=device,
    )
    model = getattr(predictor, "model", predictor)
    hooks, hooked = register_hooks(model)
    print(f"Hooked {len(hooked)} used submodules (prompt_encoder + mask_decoder, "
          "leaf modules + the two root outputs; containers/unused excluded).")

    # group pairs by image so each image is encoded once
    by_image = defaultdict(list)
    for case in shifts:
        by_image[case["image_name"]].append(case)

    # ---- accumulators ----
    # divergence pair types: critical = best vs bad, control = best vs control box
    pair_types = ["critical"] + (["control"] if args.control else [])
    acc = {pt: {"raw": defaultdict(list), "rel": defaultdict(list),
                "rms": defaultdict(list), "l1": defaultdict(list)}
           for pt in pair_types}
    # activation-std scenarios: normal = best_box, critical = bad_box, control = ctrl box
    std_scenarios = ["normal", "critical"] + (["control"] if args.control else [])
    std_acc = {sc: defaultdict(list) for sc in std_scenarios}

    layer_order: list[str] = []          # execution order (from first pair, ref=best)
    layer_group: dict[str, str] = {}
    per_case_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)

    def _register_layers(A):
        for layer in A.keys():
            if layer not in layer_group:
                layer_order.append(layer)
                layer_group[layer] = _group_of(layer)

    n_done = 0
    for image_name, cases in by_image.items():
        image_path = _find_file(args.images_dir, image_name)
        if image_path is None:
            print(f"[warn] image not found: {image_name}, skipping {len(cases)} case(s)")
            continue
        try:
            _prepare_image(str(image_path), predictor)  # encode once
        except Exception as e:
            print(f"[warn] encode failed for {image_name}: {e}")
            continue

        for case in cases:
            try:
                A_best = capture_for_box(case["best_box"], predictor, device)
                A_bad = capture_for_box(case["bad_box"], predictor, device)
                box_acts = {"normal": A_best, "critical": A_bad}
                pairs = {"critical": divergence(A_best, A_bad)}
                if args.control:
                    ctrl_box, _d = make_control_box(case["best_box"], case["bad_box"], rng)
                    A_ctrl = capture_for_box(ctrl_box, predictor, device)
                    box_acts["control"] = A_ctrl
                    pairs["control"] = divergence(A_best, A_ctrl)
            except Exception as e:
                print(f"[warn] forward failed for {image_name}: {e}")
                continue

            _register_layers(A_best)

            # divergence metrics
            for ptype, dvg in pairs.items():
                for layer, (raw, rel, rms, l1_rel) in dvg.items():
                    acc[ptype]["raw"][layer].append(raw)
                    acc[ptype]["rel"][layer].append(rel)
                    acc[ptype]["rms"][layer].append(rms)
                    acc[ptype]["l1"][layer].append(l1_rel)
                    if args.per_case_csv:
                        per_case_rows.append({
                            "image_name": image_name, "pair_type": ptype, "layer": layer,
                            "raw_l2": raw, "rel_l2": rel, "rms_l2": rms, "l1_rel": l1_rel,
                        })

            # activation std per scenario
            for sc, A in box_acts.items():
                for layer, s in activation_std(A).items():
                    std_acc[sc][layer].append(s)

            n_done += 1

    if hooks:
        for h in hooks:
            h.remove()
    if n_done == 0:
        raise SystemExit("No pairs processed (check --images_dir / JSON paths).")
    print(f"Processed {n_done} pairs over {len(by_image)} images"
          f"{' (+control)' if args.control else ''}.")

    # ---- aggregate divergence table (one row per pair_type x layer) ----
    rows = []
    for i, layer in enumerate(layer_order):
        for ptype in pair_types:
            raws = acc[ptype]["raw"].get(layer)
            if not raws:
                continue
            raw = np.array(raws)
            rel = np.array(acc[ptype]["rel"][layer])
            rms = np.array(acc[ptype]["rms"][layer])
            l1 = np.array(acc[ptype]["l1"][layer])
            rows.append({
                "order": i, "group": layer_group[layer], "layer": layer,
                "pair_type": ptype, "n": len(raw),
                "raw_l2_mean": raw.mean(), "raw_l2_std": raw.std(),
                "rel_l2_mean": rel.mean(), "rel_l2_std": rel.std(),
                "rms_l2_mean": rms.mean(), "rms_l2_std": rms.std(),
                "l1_rel_mean": l1.mean(), "l1_rel_std": l1.std(),
            })
    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved per-layer L2/L1 table -> {out_csv}  ({len(df)} rows)")

    # ---- aggregate activation-std table (wide: one row per layer) ----
    std_rows = []
    for i, layer in enumerate(layer_order):
        row = {"order": i, "group": layer_group[layer], "layer": layer}
        n_vals = []
        for sc in std_scenarios:
            vals = std_acc[sc].get(layer)
            if vals:
                v = np.array(vals)
                row[f"std_{sc}_mean"] = v.mean()
                row[f"std_{sc}_cases_std"] = v.std()  # variability across cases
                n_vals.append(len(v))
            else:
                row[f"std_{sc}_mean"] = np.nan
                row[f"std_{sc}_cases_std"] = np.nan
        row["n"] = max(n_vals) if n_vals else 0
        # convenience ratios vs the normal (best_box) baseline
        base = row.get("std_normal_mean", np.nan)
        if base and not np.isnan(base):
            row["std_critical_over_normal"] = row.get("std_critical_mean", np.nan) / base
            if "control" in std_scenarios:
                row["std_control_over_normal"] = row.get("std_control_mean", np.nan) / base
        std_rows.append(row)
    std_df = pd.DataFrame(std_rows)

    out_std_csv = Path(args.out_std_csv)
    out_std_csv.parent.mkdir(parents=True, exist_ok=True)
    std_df.to_csv(out_std_csv, index=False)
    print(f"Saved per-layer activation-std table -> {out_std_csv}  ({len(std_df)} rows)")

    if args.per_case_csv and per_case_rows:
        pc = Path(args.per_case_csv); pc.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_case_rows).to_csv(pc, index=False)
        print(f"Saved per-case L2/L1 -> {pc}")

    # ---- plots (std of the divergence = running band around each mean line) ----
    _plot_divergence(df, Path(args.out_plot))
    _plot_l1(df, Path(args.out_l1_plot))

    # quick text summary: top critical layers by relative amplification
    crit = df[df["pair_type"] == "critical"]
    top = crit.sort_values("rel_l2_mean", ascending=False).head(8)
    if args.control:
        ctrl_rel = df[df["pair_type"] == "control"].set_index("layer")["rel_l2_mean"]
        print("\nTop layers by relative L2 (critical vs control, ratio = specificity):")
        for _, r in top.iterrows():
            c = float(ctrl_rel.get(r["layer"], float("nan")))
            ratio = r["rel_l2_mean"] / c if c and not np.isnan(c) else float("nan")
            print(f"  crit {r['rel_l2_mean']:.4f} | ctrl {c:.4f} | x{ratio:.1f}  {r['layer']}")
    else:
        print("\nTop layers by relative L2 (where the pair diverges most):")
        for _, r in top.iterrows():
            print(f"  {r['rel_l2_mean']:.4f}  (raw {r['raw_l2_mean']:.3f})  {r['layer']}")


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

_STYLE = {
    "critical": dict(color="#d62728", marker="o"),
    "control":  dict(color="#555555", marker="s"),
    "normal":   dict(color="#1f77b4", marker="^"),
}


def _layer_axis(df: pd.DataFrame):
    layers = df.sort_values("order").drop_duplicates("order")[["order", "layer", "group"]]
    return layers, layers["order"].to_numpy()


def _decorate(ax, layers, x, ticks: bool = True):
    """Grid + prompt-encoder/mask-decoder separator; tick labels only where
    ticks=True (the bottom panel of a shared-x figure)."""
    ax.set_xticks(x)
    if ticks:
        ax.set_xticklabels(layers["layer"], rotation=90, fontsize=6)
    ax.grid(axis="y", ls=":", alpha=0.4)
    dec = layers[layers["group"] == "mask_decoder"]["order"]
    if len(dec) and (layers["group"] == "prompt_encoder").any():
        ax.axvline(dec.min() - 0.5, color="k", ls="--", lw=0.8, alpha=0.5)


def _draw_pair_lines(ax, df: pd.DataFrame, mean_col: str, std_col: str) -> None:
    """Mean divergence line per pair type + a running mean±std band around it
    (std across the pairs, per layer). Divergences are non-negative, so the
    lower edge of the band is clipped at 0."""
    for pt in dict.fromkeys(df["pair_type"]):
        sub = df[df["pair_type"] == pt].sort_values("order")
        style = _STYLE.get(pt, {})
        mean = sub[mean_col].to_numpy()
        std = sub[std_col].to_numpy()
        ax.plot(sub["order"], mean, label=pt, lw=1.3, ms=3.5, **style)
        ax.fill_between(sub["order"], np.maximum(mean - std, 0.0), mean + std,
                        color=style.get("color", "#888888"), alpha=0.15, lw=0)
    ax.legend(loc="upper left", title="pair (line=mean, band=±std)")


def _plot_divergence(df: pd.DataFrame, out_path: Path) -> None:
    """Raw + relative L2 vs layer, each curve with its mean±std band,
    both panels sharing the same layer axis."""
    layers, x = _layer_axis(df)
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=(max(10, 0.28 * len(layers)), 11))

    for ax, mean_col, std_col, title in (
        (axes[0], "raw_l2_mean", "raw_l2_std",
         "Raw L2  ||A_best - A_other||  (per layer, mean ± std over pairs)"),
        (axes[1], "rel_l2_mean", "rel_l2_std",
         "Relative L2  ||A_best - A_other|| / ||A_best||  (mean ± std over pairs)"),
    ):
        _draw_pair_lines(ax, df, mean_col, std_col)
        ax.set_title(title)
        _decorate(ax, layers, x, ticks=(ax is axes[-1]))

    sub_t = "critical vs control" if df["pair_type"].nunique() > 1 else "critical pairs"
    fig.suptitle(f"SAM activation divergence across used layers ({sub_t})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved L2 plot (mean ± std bands) -> {out_path}")


def _plot_l1(df: pd.DataFrame, out_path: Path) -> None:
    """Relative L1 vs layer with its mean±std band."""
    layers, x = _layer_axis(df)
    fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.28 * len(layers)), 6))

    _draw_pair_lines(ax, df, "l1_rel_mean", "l1_rel_std")
    ax.set_title("Relative L1  ||A_best - A_other||_1 / ||A_best||_1  "
                 "(per layer, mean ± std over pairs)")
    _decorate(ax, layers, x, ticks=True)

    sub_t = "critical vs control" if df["pair_type"].nunique() > 1 else "critical pairs"
    fig.suptitle(f"SAM normalised-L1 activation divergence ({sub_t})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved L1 plot (mean ± std band) -> {out_path}")


if __name__ == "__main__":
    main()
