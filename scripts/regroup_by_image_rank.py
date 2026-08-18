"""
regroup_by_image_rank.py
========================
Re-stratify an existing refine_box_iou_grad run WITHOUT re-running the models.

The buckets printed by refine_box_iou_grad ("weak <0.5 / mid / strong >=0.8")
are cut on ABSOLUTE ``undefended_iou``, which is SAM's token-0 output
(multimask_output=False, see heatmaps/defend_critical_shifts._predict_single_box).
That makes the buckets model-dependent in a way that breaks cross-model
comparison: for SAM1 vit_b, token-0 collapses on ambiguous boxes, so "weak"
selects "cases where token-0 misfired" (n=2355) rather than "cases that are
genuinely hard" (n~1740 for SAM2.1/SAM3) -- and any method that reads the 3
multimask heads recovers them for free.

This script instead ranks cases WITHIN EACH IMAGE by --rank_col and cuts them
at fixed FRACTIONS (default: worst 20% / middle 60% / best 20%).  Every image
then contributes the same proportion to every group, for every model, so the
groups are comparable across models even when the models disagree about what
is hard.  Group membership is fixed by --rank_col and then used to slice every
other IoU column -- same convention as scripts/group_robustness_deciles.py and
refine_box_iou_grad._assign_deciles, just with unequal group sizes allowed.

Choice of --rank_col (this is the whole point, so it is a flag):
  headsel_iou  (default) -- multimask head-select on bad_box, i.e. the actual
      starting point of the gradient.  Removes the token-0 bias.  Caveat: the
      ranking column is then the same measurement grad starts from, so
      regression-to-the-mean flatters grad in the weak group -- equally for
      every model, which is what makes the CROSS-MODEL comparison fair, but do
      not read the absolute weak-group lift as a pure method effect.
  undef_dist_best (+ --rank_desc) -- corner-L2 in 1024px from the attacked box
      to the reference best_box, i.e. how far the box was pushed.  Pure box
      geometry: no model touches it, so the grouping is byte-identical across
      model runs (verified on the exp_res CSVs) -- the strictest control.  It
      ranks "biggest perturbation last", hence --rank_desc to put the worst
      cases in group 1.
  clean_iou -- token-0 on the un-attacked best_box.  NOTE: in --dataset
      user_study runs there is one reference box per image, so clean_iou is
      constant within an image and CANNOT be used for within-image ranking.
  bad_iou_json / best_iou_json -- only populated by --dataset critical_shifts;
      they are NaN in user_study runs.

Usage
-----
    python scripts/regroup_by_image_rank.py --csv "exp_res/*_m.csv"

    # strict control: group by how far the box was pushed, worst first
    python scripts/regroup_by_image_rank.py --csv "exp_res/*_m.csv" \
        --rank_col undef_dist_best --rank_desc

    # 10 equal groups (deciles)
    python scripts/regroup_by_image_rank.py --csv "exp_res/*_m.csv" \
        --fractions 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1

    # equal weight per image instead of per case
    python scripts/regroup_by_image_rank.py --csv "exp_res/*_m.csv" --per_image_mean
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# IoU columns reported per group, in print order. Missing ones are skipped
# (a --grad_only / --fast run does not write every method).
METHOD_COLS = [
    ("undef", "undefended_iou"),
    ("headsel", "headsel_iou"),
    ("bon", "best_of_n_iou"),
    ("grad", "grad_final_iou"),
    ("gradBest", "grad_best_iou"),
]


def assign_fraction_groups(
    df: pd.DataFrame,
    image_col: str,
    rank_col: str,
    fractions: np.ndarray,
    descending: bool = False,
) -> pd.Series:
    """Per image: sort rows by rank_col (ascending unless `descending`), cut at
    cumulative `fractions` of that image's case count. Returns a 1-indexed
    group id (1 = worst, i.e. the low end of rank_col, or the high end when
    `descending`). Rows with NaN rank_col stay NaN.

    Boundaries are rounded, not floored, and the last group always runs to the
    end -- so the groups tile the image exactly even when n_cases * fraction is
    not an integer (25 cases at 0.2/0.6/0.2 -> 5/15/5).
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = df[rank_col].notna()
    cum = np.cumsum(fractions)
    for _, g in df[valid].groupby(df.loc[valid, image_col], sort=False):
        order = g[rank_col].sort_values(kind="mergesort", ascending=not descending).index
        n = len(order)
        edges = np.rint(cum * n).astype(int)
        edges = np.clip(edges, 0, n)
        edges[-1] = n
        start = 0
        for gi, end in enumerate(edges, start=1):
            if end > start:
                out.loc[order[start:end]] = gi
            start = max(start, end)
    return out


def group_table(
    df: pd.DataFrame,
    group_col: str,
    names: list[str],
    cols: list[tuple[str, str]],
    per_image_mean: bool,
) -> pd.DataFrame:
    """Mean of every method column per group. per_image_mean: average within
    each image first, so every image weighs the same regardless of how many
    cases it contributed (matches group_robustness_deciles.py)."""
    rows = []
    for gi, name in enumerate(names, start=1):
        sub = df[df[group_col] == gi]
        if sub.empty:
            continue
        rec = {"group": name, "n": len(sub)}
        for label, col in cols:
            if per_image_mean:
                rec[label] = sub.groupby("image_name")[col].mean().mean()
            else:
                rec[label] = sub[col].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def _fmt_table(tab: pd.DataFrame, cols: list[tuple[str, str]]) -> str:
    """Group table + the two deltas that matter: grad over the token-0 baseline
    (what the original report printed) and grad over head-select (the honest
    gradient effect, since head-select is where the gradient starts)."""
    labels = [lab for lab, _ in cols]
    head = f"  {'group':>8} | {'n':>5} | " + " | ".join(f"{l:>8}" for l in labels)
    extra = []
    if "grad" in labels and "undef" in labels:
        extra.append("gradVSundef")
    if "grad" in labels and "headsel" in labels:
        extra.append("gradVShsel")
    head += " | " + " | ".join(f"{e:>11}" for e in extra) if extra else ""
    lines = [head, "  " + "-" * (len(head) - 2)]
    for _, r in tab.iterrows():
        line = f"  {r['group']:>8} | {int(r['n']):>5} | " + " | ".join(
            f"{r[l]:>8.3f}" for l in labels
        )
        vals = []
        if "gradVSundef" in extra:
            d = r["grad"] - r["undef"]
            vals.append(f"{d:>+7.3f} {d / r['undef'] * 100:>+5.1f}%" if r["undef"] else f"{d:>+7.3f}")
        if "gradVShsel" in extra:
            d = r["grad"] - r["headsel"]
            vals.append(f"{d:>+7.3f} {d / r['headsel'] * 100:>+5.1f}%" if r["headsel"] else f"{d:>+7.3f}")
        if vals:
            line += " | " + " | ".join(vals)
        lines.append(line)
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", nargs="+", required=True,
                   help="per-case CSVs from refine_box_iou_grad (--out_csv); "
                        "globs allowed. One model per file; the file stem is "
                        "used as the model label.")
    p.add_argument("--rank_col", default="headsel_iou",
                   help="column the within-image ranking is built on "
                        "(default: headsel_iou -- see module docstring for the "
                        "trade-offs of undef_dist_best / clean_iou)")
    p.add_argument("--rank_desc", action="store_true",
                   help="rank descending, i.e. the HIGHEST rank_col is the "
                        "worst -- use for distance-like columns such as "
                        "undef_dist_best, where bigger means more perturbed")
    p.add_argument("--fractions", default="0.2,0.6,0.2",
                   help="comma-separated group sizes as fractions of each "
                        "image's cases, worst-first (default: 0.2,0.6,0.2)")
    p.add_argument("--names", default=None,
                   help="comma-separated group names (default: weak,mid,strong "
                        "for 3 groups, G1..Gk otherwise)")
    p.add_argument("--per_image_mean", action="store_true",
                   help="average per image first, then over images (equal "
                        "weight per image instead of per case)")
    p.add_argument("--out_csv", default=None,
                   help="also write the long-form table (model,group,...) here")
    return p.parse_args()


def main():
    args = parse_args()

    fractions = np.array([float(x) for x in args.fractions.split(",")], dtype=float)
    if (fractions <= 0).any():
        raise SystemExit("--fractions must all be > 0")
    total = fractions.sum()
    if not np.isclose(total, 1.0):
        print(f"[warn] --fractions sum to {total:.3f}, normalising to 1.0")
        fractions = fractions / total
    k = len(fractions)

    if args.names:
        names = [s.strip() for s in args.names.split(",")]
        if len(names) != k:
            raise SystemExit(f"--names has {len(names)} entries, --fractions has {k}")
    elif k == 3:
        names = ["weak", "mid", "strong"]
    else:
        names = [f"G{i}" for i in range(1, k + 1)]

    paths = [Path(p) for pat in args.csv for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no CSVs matched {args.csv}")

    pct = ", ".join(f"{n} {f * 100:.0f}%" for n, f in zip(names, fractions))
    direction = "highest-first" if args.rank_desc else "lowest-first"
    print(f"Ranking within each image by '{args.rank_col}' ({direction} = group 1); "
          f"groups: {pct}")
    print(f"Weighting: {'equal per image' if args.per_image_mean else 'equal per case'}\n")

    all_tabs = []
    for path in paths:
        df = pd.read_csv(path)
        label = path.stem
        if args.rank_col not in df.columns:
            print(f"[skip] {label}: no column '{args.rank_col}'")
            continue
        cols = [(lab, c) for lab, c in METHOD_COLS
                if c in df.columns and df[c].notna().any()]
        if not cols:
            print(f"[skip] {label}: no method IoU columns")
            continue

        if df[args.rank_col].isna().all():
            print(f"[skip] {label}: '{args.rank_col}' is entirely NaN "
                  f"(bad_iou_json/best_iou_json are only written by "
                  f"--dataset critical_shifts)")
            continue
        within = df.groupby("image_name")[args.rank_col].std().mean()
        if not within > 0:
            print(f"[skip] {label}: '{args.rank_col}' is constant within every "
                  f"image, so it cannot rank cases inside an image")
            continue

        df["_grp"] = assign_fraction_groups(df, "image_name", args.rank_col,
                                            fractions, args.rank_desc)
        dropped = int(df["_grp"].isna().sum())
        tab = group_table(df, "_grp", names, cols, args.per_image_mean)

        note = f"  ({dropped} cases dropped: NaN {args.rank_col})" if dropped else ""
        print(f"=== {label} === {len(df)} cases / {df['image_name'].nunique()} images{note}")
        print(_fmt_table(tab, cols))
        print()

        tab.insert(0, "model", label)
        all_tabs.append(tab)

    all_tabs = [t for t in all_tabs if not t.empty]
    if not all_tabs:
        raise SystemExit(f"nothing to report -- no usable rows for "
                         f"--rank_col {args.rank_col}")

    combined = pd.concat(all_tabs, ignore_index=True)

    # cross-model pivot: one block per group, models as rows -- this is the
    # view that makes "is model X's weak-group lift real or an artefact of its
    # own baseline" readable at a glance.
    if len(all_tabs) > 1:
        labels = [c for c in ["undef", "headsel", "bon", "grad", "gradBest"]
                  if c in combined.columns]
        for name in names:
            sub = combined[combined["group"] == name]
            if sub.empty:
                continue
            print(f"--- group '{name}' across models ---")
            width = max(len(m) for m in sub["model"])
            head = f"  {'model':>{width}} | {'n':>5} | " + " | ".join(f"{l:>8}" for l in labels)
            print(head)
            print("  " + "-" * (len(head) - 2))
            for _, r in sub.iterrows():
                print(f"  {r['model']:>{width}} | {int(r['n']):>5} | "
                      + " | ".join(f"{r[l]:>8.3f}" for l in labels))
            print()

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
