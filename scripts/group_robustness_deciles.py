"""
group_robustness_deciles.py
===========================
Decile (10%-group) robustness analysis on the user-study dataset.

For every IMAGE independently we:
    1. take all user-drawn bounding boxes for that image,
    2. sort them by ``undefended_iou`` ascending (worst boxes first),
    3. split the sorted list into 10 equal-ish groups with ``np.array_split``
       (G1 = worst 10%, G2 = 10-20%, ... , G10 = best 10%),
    4. inside EACH group compute mean and std of the IoU
       (for BOTH the undefended and the defended / smoothed prediction).

The group membership is FIXED by the undefended IoU, so the same set of
boxes is used to report undefended *and* defended stats -- this lets you
read off directly how much Randomized-Smoothing lifts the worst boxes
(e.g. compare ``G1_undef_iou_mean`` vs ``G1_def_iou_mean``).

The per-image (mean, std) of every group are then AVERAGED over all images
(equal weight per image), yielding the final table:

    group | undef_iou_mean | undef_iou_std | def_iou_mean | def_iou_std | iou_recovery
    G1    |      ...        |      ...       |     ...      |     ...      |    ...
    ...
    G10   |      ...        |      ...       |     ...      |     ...      |    ...

------------------------------------------------------------------------------
Per-box IoUs come from ``heatmaps/defend_user_study.py`` (which is where the
smoothing/averaging logic lives).  Two ways to feed them in:

A) Re-use an existing CSV (recommended for big / multi-GPU runs)
   First produce per-box IoUs once (optionally multi-GPU via torchrun):
       torchrun --nproc_per_node=8 heatmaps/defend_user_study.py \
           --root ../user_study --checkpoint_path /.../sam_vit_b.pth \
           --use mp --Y 16 --sigma 0.05 --averaging_mode sigmoid \
           --output_csv outputs_smoothed/user_study_defence.csv
   torchrun writes one CSV per rank: user_study_defence.rank0.csv, .rank1.csv ...
   Then run THIS script on those shards (globs / multiple paths allowed):
       python scripts/group_robustness_deciles.py \
           --from_csv "outputs_smoothed/user_study_defence.rank*.csv" \
           --out_csv  outputs_smoothed/decile_robustness.csv

B) Compute on the fly (single GPU) -- this script calls defend_user_study.run()
       python scripts/group_robustness_deciles.py \
           --root ../user_study \
           --checkpoint_path /.../sam_vit_b_01ec64.pth \
           --model_name SAM --model_type vit_b \
           --use mp --Y 16 --sigma 0.05 --averaging_mode sigmoid \
           --perbox_csv outputs_smoothed/user_study_defence.csv \
           --out_csv    outputs_smoothed/decile_robustness.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Repo root on sys.path so that ``heatmaps`` is importable as a package
# (defend_user_study itself uses absolute ``from heatmaps...`` imports).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Step 1 -- obtain per-box IoUs  (either read CSV, or compute via run())
# ---------------------------------------------------------------------------

def _normalize_use(use: str) -> str:
    """Mirror defend_user_study.parse_args() normalisation of --use."""
    use = "".join(sorted(set(use.lower())))
    for ch in use:
        if ch not in ("p", "m"):
            raise SystemExit(f"--use must be a subset of 'pm', got {use!r}")
    return use


def load_perbox_from_csv(patterns: list[str]) -> pd.DataFrame:
    """Read & concatenate one or more defend_user_study output CSVs.

    Accepts explicit paths and/or glob patterns (e.g. rank-sharded files
    produced by torchrun: ``user_study_defence.rank*.csv``).
    """
    paths: list[str] = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if matched:
            paths.extend(matched)
        elif Path(pat).exists():
            paths.append(pat)
        else:
            print(f"[warn] --from_csv pattern matched nothing: {pat!r}")

    if not paths:
        raise SystemExit("No CSV files found for --from_csv.")

    print(f"Reading {len(paths)} CSV file(s):")
    frames = []
    for p in paths:
        print(f"  - {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} per-box rows.")
    return df


def compute_perbox_via_run(args) -> pd.DataFrame:
    """Call heatmaps.defend_user_study.run() to compute per-box IoUs.

    Heavy deps (torch, segment_anything) are imported lazily here so that the
    --from_csv analysis mode works on a machine without the model installed.
    """
    from heatmaps.defend_user_study import run as run_user_study_defence

    if not args.checkpoint_path:
        raise SystemExit("--checkpoint_path is required when not using --from_csv.")

    # Build the Namespace defend_user_study.run() expects (same field names
    # as its own parse_args()).  We do NOT set LOCAL_RANK -> world_size==1.
    ns = SimpleNamespace(
        root=args.root,
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        model_type=args.model_type,
        gpu=args.gpu,
        use=_normalize_use(args.use),
        limit_images=args.limit_images,
        Y=args.Y,
        sigma=args.sigma,
        sigma_center=args.sigma_center,
        perturb_mode=args.perturb_mode,
        averaging_mode=args.averaging_mode,
        include_base_box=args.include_base_box,
        seed=args.seed,
        output_csv=args.perbox_csv,
        topk=args.topk,
        stability_M=args.stability_M,
        stability_sigma=args.stability_sigma,
    )
    print("Computing per-box IoUs via defend_user_study.run() ...")
    df = run_user_study_defence(ns)
    print(f"defend_user_study produced {len(df)} per-box rows "
          f"(also saved to {args.perbox_csv}).")
    return df


# ---------------------------------------------------------------------------
# Step 2 -- decile grouping + aggregation
# ---------------------------------------------------------------------------

def per_image_group_stats(
    df: pd.DataFrame,
    image_col: str,
    rank_col: str,
    metric_cols: list[str],
    n_groups: int,
    min_boxes: int,
    std_ddof: int,
) -> pd.DataFrame:
    """For each image: sort by rank_col, split into n_groups, compute
    mean/std of every metric inside each group.

    Returns a long DataFrame with one row per (image, group).
    """
    needed = {image_col, rank_col, *metric_cols}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(
            f"Input is missing required column(s): {sorted(missing)}.\n"
            f"Available columns: {sorted(df.columns)}"
        )

    # drop rows with NaN in any column we rely on
    df = df.dropna(subset=list(needed)).copy()

    records: list[dict] = []
    n_skipped = 0
    for stem, g in df.groupby(image_col):
        if len(g) < min_boxes:
            n_skipped += 1
            continue

        g_sorted = g.sort_values(rank_col, ascending=True, kind="mergesort")
        # split the row positions into n_groups contiguous chunks
        chunks = np.array_split(np.arange(len(g_sorted)), n_groups)

        for gi, idx in enumerate(chunks, start=1):
            sub = g_sorted.iloc[idx]
            rec: dict = {
                image_col: stem,
                "group": gi,
                "n_boxes": int(len(sub)),
            }
            for m in metric_cols:
                vals = sub[m].to_numpy(dtype=float)
                rec[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
                # ddof=0 (population) by default -> size-1 groups give std 0
                # instead of NaN, so they still contribute to the average.
                rec[f"{m}_std"] = (
                    float(np.std(vals, ddof=std_ddof)) if len(vals) else np.nan
                )
            records.append(rec)

    if n_skipped:
        print(f"[info] skipped {n_skipped} image(s) with < {min_boxes} boxes.")

    if not records:
        raise SystemExit(
            f"No image had >= {min_boxes} boxes; nothing to aggregate. "
            f"Lower --min_boxes if this is unexpected."
        )

    return pd.DataFrame.from_records(records)


def aggregate_over_images(
    per_image: pd.DataFrame,
    metric_cols: list[str],
    std_ddof: int,
) -> pd.DataFrame:
    """Average the per-image (mean, std) of each group across all images.

    Final stat columns are named ``G{g}`` style via the 'group' index, e.g.
    undefended_iou_mean -> mean over images of the per-image group mean.
    """
    stat_cols = []
    for m in metric_cols:
        stat_cols += [f"{m}_mean", f"{m}_std"]

    agg = (
        per_image
        .groupby("group")[stat_cols]
        .mean()                      # average over images (equal weight/image)
        .reset_index()
    )

    # how many images contributed to each group + typical group size
    extra = (
        per_image
        .groupby("group")
        .agg(n_images=("group", "size"), avg_group_size=("n_boxes", "mean"))
        .reset_index()
    )
    agg = agg.merge(extra, on="group")

    # convenience: IoU recovery of the defended over the undefended group mean
    if "undefended_iou_mean" in agg.columns and "defended_iou_mean" in agg.columns:
        agg["iou_recovery"] = agg["defended_iou_mean"] - agg["undefended_iou_mean"]

    agg["group_label"] = agg["group"].map(lambda g: f"G{g}")
    return agg.sort_values("group").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_table(agg: pd.DataFrame, metric_cols: list[str], rank_col: str) -> None:
    print("\n" + "=" * 78)
    print(f"Decile robustness  (groups fixed by '{rank_col}', worst -> best; "
          f"averaged over {int(agg['n_images'].max())} images)")
    print("=" * 78)

    header = f"{'grp':<4}{'n_img':>6}{'size':>6}"
    for m in metric_cols:
        header += f"{m + '_mean':>20}{m + '_std':>20}"
    if "iou_recovery" in agg.columns:
        header += f"{'recovery':>12}"
    print(header)
    print("-" * len(header))

    for _, r in agg.iterrows():
        line = f"{r['group_label']:<4}{int(r['n_images']):>6}{r['avg_group_size']:>6.1f}"
        for m in metric_cols:
            line += f"{r[f'{m}_mean']:>20.4f}{r[f'{m}_std']:>20.4f}"
        if "iou_recovery" in agg.columns:
            line += f"{r['iou_recovery']:>+12.4f}"
        print(line)
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ----- source of per-box IoUs -----
    src = p.add_argument_group("per-box IoU source")
    src.add_argument(
        "--from_csv", nargs="+", default=None,
        help="One or more defend_user_study output CSVs (paths or globs, e.g. "
             "'out.rank*.csv'). If given, skip inference and just analyse them.",
    )

    # ----- inference args (only used when --from_csv is NOT given) -----
    inf = p.add_argument_group("inference (used only without --from_csv)")
    inf.add_argument("--root", help="user_study root (images/, masks/, user_masks/)")
    inf.add_argument("--checkpoint_path")
    inf.add_argument("--model_name", default="SAM",
                     choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    inf.add_argument("--model_type", default="vit_b")
    inf.add_argument("--gpu", type=int, default=0)
    inf.add_argument("--use", default="mp", help="user_mask kinds: 'm', 'p' or 'mp' (default: both)")
    inf.add_argument("--limit_images", type=int, default=0)
    inf.add_argument("--Y", type=int, default=16)
    inf.add_argument("--sigma", type=float, default=0.05)
    inf.add_argument("--sigma_center", type=float, default=0.03)
    inf.add_argument("--perturb_mode", default="size", choices=["size", "size_center"])
    inf.add_argument("--averaging_mode", default="sigmoid",
                     choices=["logit", "sigmoid", "binary", "score_weighted",
                              "best_of_n", "saif_score", "saif_stability"])
    inf.add_argument("--topk", type=int, default=3,
                     help="SAIF modes: keep top-k most confident candidate boxes")
    inf.add_argument("--stability_M", type=int, default=6,
                     help="saif_stability: micro-perturbations per candidate")
    inf.add_argument("--stability_sigma", type=float, default=None,
                     help="saif_stability: micro-perturbation sigma (default=--sigma)")
    inf.add_argument("--include_base_box", action="store_true", default=False)
    inf.add_argument("--seed", type=int, default=42)
    inf.add_argument("--perbox_csv", default="outputs_smoothed/user_study_defence.csv",
                     help="where defend_user_study.run() saves its per-box CSV")

    # ----- decile analysis -----
    an = p.add_argument_group("decile analysis")
    an.add_argument("--image_col", default="image_stem",
                    help="column identifying the image")
    an.add_argument("--rank_col", default="undefended_iou",
                    help="column used to sort boxes and fix the 10%% groups")
    an.add_argument("--metric_cols", nargs="+",
                    default=["undefended_iou", "defended_iou"],
                    help="IoU columns to summarise (mean/std) inside each group")
    an.add_argument("--n_groups", type=int, default=10)
    an.add_argument("--min_boxes", type=int, default=10,
                    help="skip images with fewer than this many user boxes")
    an.add_argument("--std_ddof", type=int, default=0,
                    help="ddof for std (0=population; 1=sample, NaN for size-1 groups)")

    # ----- output -----
    p.add_argument("--out_csv", default="outputs_smoothed/decile_robustness.csv",
                   help="summary table (one row per group)")
    p.add_argument("--per_image_csv", default=None,
                   help="optional: also save the per-image/per-group breakdown")

    return p.parse_args()


def main():
    args = parse_args()

    # 1) per-box IoUs
    if args.from_csv:
        df = load_perbox_from_csv(args.from_csv)
    else:
        if not args.root:
            raise SystemExit("Provide either --from_csv or --root (+ --checkpoint_path).")
        # SAM-HQ2/SAM3 need a different conda env (see heatmaps/env_dispatch.py);
        # re-exec there before any torch/segment_anything import happens.
        from heatmaps.env_dispatch import maybe_dispatch_to_env
        maybe_dispatch_to_env(args.model_name, __file__)
        df = compute_perbox_via_run(args)

    # 2) per-image decile stats
    per_image = per_image_group_stats(
        df,
        image_col=args.image_col,
        rank_col=args.rank_col,
        metric_cols=args.metric_cols,
        n_groups=args.n_groups,
        min_boxes=args.min_boxes,
        std_ddof=args.std_ddof,
    )

    # 3) average over images
    agg = aggregate_over_images(per_image, args.metric_cols, args.std_ddof)

    # 4) outputs
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"Saved summary -> {out_path}")

    if args.per_image_csv:
        pi_path = Path(args.per_image_csv)
        pi_path.parent.mkdir(parents=True, exist_ok=True)
        per_image.to_csv(pi_path, index=False)
        print(f"Saved per-image breakdown -> {pi_path}")

    print_table(agg, args.metric_cols, args.rank_col)


if __name__ == "__main__":
    main()
