"""
train_gate.py
=============
Train and score a GATE for the box-refinement defence: given one case, decide
whether to keep the gradient result or fall back to head-select.

Motivation. Stratified within-image (scripts/regroup_by_image_rank.py), the
gradient helps on the weak fifth of cases and HURTS on the strong fifth, for
every model measured. A per-case gate is what turns that into a net win. Its
ceiling is max(headsel, grad) per case.

The question this script answers is narrow and specific: do the tier-1
mask-level features (t1_*, added to refine_box_iou_grad) buy anything over the
scalars already available from SAM's predicted-IoU head? So it fits the same
model on three feature sets -- "pred", "tier1", "both" -- and reports each.

Target is the GAIN, not its sign. Roughly half the cases are near-ties whose
sign is a coin flip and whose cost is nil; squared loss on the gain weighs each
case by how much the decision actually matters, and the operating threshold can
then be tuned after the fact.

Two honesty constraints, both easy to get wrong here:
  * Validation is GroupKFold on image_name. The 25 cases of one image share an
    image and an object; a random split leaks between train and test.
  * Features are WHITELISTED, never blacklisted. Every true-IoU column, every
    clean_* column and every *_dist_best column is derived from the ground
    truth or from the reference box -- those are the target, not features.

    python scripts/train_gate.py --csv results/sam3_tier1_50img.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

# Scalars available at inference from the predicted-IoU head and from how far
# each method moved the box. This is the baseline the tier-1 block must beat.
PRED_COLS = [
    "undef_pred", "headsel_pred", "bon_pred", "grad_final_pred",
    "grad_corner_l2", "grad_center_shift", "grad_w_delta", "grad_h_delta",
    "bon_corner_l2", "bon_center_shift", "bon_w_delta", "bon_h_delta",
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Whitelisted feature matrix plus the column index of each feature set."""
    X = pd.DataFrame(index=df.index)
    pred = [c for c in PRED_COLS if c in df.columns and df[c].notna().any()]
    for c in pred:
        X[c] = df[c]
    # differences a tree cannot form on its own from the raw scores
    if {"grad_final_pred", "headsel_pred"} <= set(df.columns):
        X["d_pred_grad_hsel"] = df.grad_final_pred - df.headsel_pred
        pred.append("d_pred_grad_hsel")
    if {"bon_pred", "undef_pred"} <= set(df.columns):
        X["d_pred_bon_undef"] = df.bon_pred - df.undef_pred
        pred.append("d_pred_bon_undef")
    tier1 = [c for c in df.columns if c.startswith("t1_") and df[c].notna().any()]
    for c in tier1:
        X[c] = df[c]
    return X, {"pred": pred, "tier1": tier1, "both": pred + tier1}


def oof_predict(X, y, groups, n_splits, seed):
    """Out-of-fold predictions: every row is scored by a model that never saw
    that row's IMAGE."""
    from catboost import CatBoostRegressor
    oof = np.zeros(len(y))
    imps = []
    n_splits = min(n_splits, groups.nunique())
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                              loss_function="RMSE", random_seed=seed, verbose=0)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
        imps.append(pd.Series(m.get_feature_importance(), index=X.columns))
    return oof, pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


def policy_report(name, oof, df, thr):
    """Value of the gate as a POLICY, not as a regression.

    The comparison against plain grad is paired -- the two differ only on the
    cases the gate rejects -- so the standard error is clustered by image over
    that paired difference, which is far tighter than the spread of the gain.
    """
    h, g = df.headsel_iou.values, df.grad_final_iou.values
    gated = np.where(oof > thr, g, h)
    diff = pd.Series(gated - g)
    per_img = diff.groupby(df.image_name.values).mean()
    se = per_img.std() / np.sqrt(per_img.size)
    return {
        "features": name, "rho": spearmanr(oof, g - h).statistic,
        "thr": thr, "kept_grad": float((oof > thr).mean()),
        "gated": gated.mean(), "vs_grad": diff.mean(),
        "se": se, "z": abs(diff.mean()) / se if se > 0 else np.nan,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True,
                   help="per-case CSV(s) from refine_box_iou_grad; globs allowed. "
                        "Several files are pooled into one training set.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thresholds", default="0.0,0.002,0.005,0.01",
                   help="gate operating points to report (predicted gain above "
                        "which the gradient result is kept)")
    p.add_argument("--out_csv", default=None)
    a = p.parse_args()

    paths = [Path(q) for pat in a.csv for q in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no CSVs matched {a.csv}")
    frames = []
    for q in paths:
        f = pd.read_csv(q)
        f["model"] = q.stem
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    if not any(c.startswith("t1_") for c in df.columns):
        raise SystemExit("no t1_* columns -- this CSV predates the tier-1 features")

    y = df.grad_final_iou - df.headsel_iou
    groups = df.image_name
    X, sets = build_features(df)

    print(f"{len(df)} cases / {groups.nunique()} images / {len(paths)} run(s)")
    print(f"gain grad-headsel: mean {y.mean():+.4f} | grad better in "
          f"{(y > 0).mean():.1%} | ties |g|<0.01 in {(y.abs() < 0.01).mean():.1%}")
    h, g = df.headsel_iou.values, df.grad_final_iou.values
    orc = np.maximum(h, g).mean()
    print(f"headsel {h.mean():.4f} | grad {g.mean():.4f} | ORACLE gate {orc:.4f} "
          f"(+{orc - g.mean():.4f} over grad)\n")

    thrs = [float(t) for t in a.thresholds.split(",")]
    rows, imps = [], {}
    for name in ("pred", "tier1", "both"):
        cols = sets[name]
        if not cols:
            continue
        oof, imp = oof_predict(X[cols], y, groups, a.folds, a.seed)
        imps[name] = imp
        for t in thrs:
            rows.append(policy_report(name, oof, df, t))
    r = pd.DataFrame(rows)

    print(f"{'features':<8} | {'thr':>6} | {'rho':>6} | {'keep%':>6} | "
          f"{'gated IoU':>9} | {'vs grad':>8} | {'z':>5}")
    print("-" * 68)
    for _, x in r.iterrows():
        print(f"{x.features:<8} | {x.thr:>6.3f} | {x.rho:>6.3f} | "
              f"{x.kept_grad:>6.1%} | {x.gated:>9.4f} | {x.vs_grad:>+8.4f} | {x.z:>5.1f}")

    if "both" in imps:
        print("\ntop features (set 'both'):")
        print(imps["both"].head(15).to_string())
        share = imps["both"].filter(like="t1_").sum() / imps["both"].sum()
        print(f"\ntier-1 share of total importance: {share:.1%}")

    if a.out_csv:
        Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
        r.to_csv(a.out_csv, index=False)
        print(f"\nSaved -> {a.out_csv}")


if __name__ == "__main__":
    main()
