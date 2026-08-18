"""
train_gate.py
=============
Train and score a GUARD for the box-refinement defence: per case, decide
whether to keep the gradient-refined mask or revert to the original one.

Motivation. Stratified within-image (scripts/regroup_by_image_rank.py), the
gradient helps on the weak fifth of cases and HURTS on the strong fifth, for
every model measured. A per-case guard is what turns that into a net win. Its
ceiling is max(headsel, grad) per case.

The guard reverts to HEAD-SELECT -- the gradient's own starting point, i.e.
"the ascent never happened" -- not to token-0, which is a different (and for
SAM1 much worse) mask.

Feature sets are reported separately so the question "do the new features buy
anything over SAM's own predicted-IoU head?" gets a direct answer:
    pred  -- scalars from the predicted-IoU head + absolute box displacement
    t1    -- tier-1: mask-level agreement, head agreement, mask plausibility
    t2    -- tier-2: relative box geometry, box priors, trajectory shape
    all   -- everything

Target is the signed GAIN, not its sign: roughly half the cases are near-ties
whose sign is a coin flip and whose cost is nil, and squared loss weighs each
case by how much the decision actually matters.

Two honesty constraints, both easy to get wrong here:
  * Splits are BY IMAGE. The 25 cases of one image share an image and an
    object, so a random split leaks between train and test.
  * Features are WHITELISTED, never blacklisted. Every true-IoU column, every
    clean_* column and every *_dist_best column is derived from the ground
    truth or from the reference box -- those are the target, not features.
  * The operating threshold is tuned on TRAIN only (out-of-fold), then applied
    once to the held-out images.

    python scripts/train_gate.py --csv results/sam3_full_t12.csv \\
        --holdout_images 100
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
# each method moved the box. This is the baseline the new blocks must beat.
PRED_COLS = [
    "undef_pred", "headsel_pred", "bon_pred", "grad_final_pred",
    "grad_corner_l2", "grad_center_shift", "grad_w_delta", "grad_h_delta",
    "bon_corner_l2", "bon_center_shift", "bon_w_delta", "bon_h_delta",
]
GROUP_FRACS = (0.2, 0.6, 0.2)
GROUP_NAMES = ("weak", "mid", "strong")


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
    t1 = [c for c in df.columns if c.startswith("t1_") and df[c].notna().any()]
    t2 = [c for c in df.columns if c.startswith("t2_") and df[c].notna().any()]
    for c in t1 + t2:
        X[c] = df[c]
    return X, {"pred": pred, "t1": t1, "t2": t2, "all": pred + t1 + t2}


def within_image_groups(df: pd.DataFrame, rank_col: str = "headsel_iou") -> np.ndarray:
    """Rank each image's cases by rank_col and cut at GROUP_FRACS, so every
    image contributes the same proportion to every group."""
    out = np.empty(len(df), dtype=object)
    cum = np.cumsum(GROUP_FRACS)
    for _, g in df.groupby("image_name", sort=False):
        order = g[rank_col].sort_values(kind="mergesort").index
        n = len(order)
        edges = np.clip(np.rint(cum * n).astype(int), 0, n)
        edges[-1] = n
        start = 0
        for name, end in zip(GROUP_NAMES, edges):
            if end > start:
                out[df.index.get_indexer(order[start:end])] = name
            start = max(start, end)
    return out


def _fit_predict(Xtr, ytr, Xte, seed):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                          loss_function="RMSE", random_seed=seed, verbose=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte), pd.Series(m.get_feature_importance(), index=Xtr.columns)


def oof_predict(X, y, groups, n_splits, seed):
    """Out-of-fold predictions: every row is scored by a model that never saw
    that row's IMAGE."""
    oof = np.zeros(len(y))
    imps = []
    n_splits = min(n_splits, groups.nunique())
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pred, imp = _fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[te], seed)
        oof[te] = pred
        imps.append(imp)
    return oof, pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


def guard_metrics(score, df, thr):
    """Value of the guard as a POLICY, not as a regression.

    guarded-vs-grad is a PAIRED difference (they differ only on the cases the
    guard reverts), so the standard error is clustered by image over that
    difference -- far tighter than the spread of the gain itself.
    """
    h, g = df.headsel_iou.values, df.grad_final_iou.values
    guarded = np.where(score > thr, g, h)
    diff = pd.Series(guarded - g)
    per_img = diff.groupby(df.image_name.values).mean()
    se = per_img.std() / np.sqrt(max(1, per_img.size))
    return {
        "n": len(df), "headsel": h.mean(), "grad": g.mean(),
        "guarded": guarded.mean(), "oracle": np.maximum(h, g).mean(),
        "kept": float((score > thr).mean()), "vs_grad": diff.mean(),
        "se": se, "z": abs(diff.mean()) / se if se > 0 else np.nan,
    }


def tune_threshold(score, df, grid):
    """Pick the operating point that maximises the guarded mean. Called on
    TRAIN out-of-fold predictions only."""
    best, best_t = -np.inf, 0.0
    for t in grid:
        v = guard_metrics(score, df, t)["guarded"]
        if v > best:
            best, best_t = v, t
    return best_t


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True,
                   help="per-case CSV(s) from refine_box_iou_grad; globs allowed")
    p.add_argument("--holdout_images", type=int, default=100,
                   help="images held out for the final report; the rest train. "
                        "0 = no holdout, report cross-validated on everything")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thresholds", default=None,
                   help="comma-separated grid to tune the operating point over "
                        "(default: a spread around zero)")
    p.add_argument("--out_csv", default=None)
    a = p.parse_args()

    paths = [Path(q) for pat in a.csv for q in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no CSVs matched {a.csv}")
    df = pd.concat([pd.read_csv(q).assign(model=q.stem) for q in paths],
                   ignore_index=True)
    if not any(c.startswith("t1_") for c in df.columns):
        raise SystemExit("no t1_* columns -- this CSV predates the gate features")

    y = df.grad_final_iou - df.headsel_iou
    X, sets = build_features(df)
    grid = ([float(t) for t in a.thresholds.split(",")] if a.thresholds
            else list(np.round(np.arange(-0.02, 0.0201, 0.0025), 4)))

    h, g = df.headsel_iou.values, df.grad_final_iou.values
    print(f"{len(df)} cases / {df.image_name.nunique()} images / {len(paths)} run(s)")
    print(f"features: pred={len(sets['pred'])} t1={len(sets['t1'])} "
          f"t2={len(sets['t2'])} all={len(sets['all'])}")
    print(f"gain grad-headsel: mean {y.mean():+.4f} | grad better in "
          f"{(y > 0).mean():.1%} | ties |g|<0.01 in {(y.abs() < 0.01).mean():.1%}")
    print(f"headsel {h.mean():.4f} | grad {g.mean():.4f} | "
          f"ORACLE guard {np.maximum(h, g).mean():.4f}\n")

    imgs = np.sort(df.image_name.unique())
    if a.holdout_images > 0:
        if a.holdout_images >= len(imgs):
            raise SystemExit(f"--holdout_images {a.holdout_images} >= {len(imgs)} images")
        rng = np.random.default_rng(a.split_seed)
        test_imgs = set(rng.choice(imgs, size=a.holdout_images, replace=False))
        te_mask = df.image_name.isin(test_imgs).values
        print(f"split by image: train {(~te_mask).sum()} cases / "
              f"{len(imgs) - len(test_imgs)} images | "
              f"test {te_mask.sum()} cases / {len(test_imgs)} images "
              f"(seed {a.split_seed})\n")
    else:
        te_mask = np.ones(len(df), bool)
        print("no holdout: reporting cross-validated over all images\n")

    tr_df, te_df = df[~te_mask], df[te_mask]
    rows, imps, scores = [], {}, {}
    for name in ("pred", "t1", "t2", "all"):
        cols = sets[name]
        if not cols:
            continue
        if a.holdout_images > 0:
            oof_tr, _ = oof_predict(X.loc[~te_mask, cols], y[~te_mask],
                                    tr_df.image_name, a.folds, a.seed)
            thr = tune_threshold(oof_tr, tr_df, grid)
            score, imp = _fit_predict(X.loc[~te_mask, cols], y[~te_mask],
                                      X.loc[te_mask, cols], a.seed)
        else:
            score, imp = oof_predict(X[cols], y, df.image_name, a.folds, a.seed)
            thr = tune_threshold(score, df, grid)
        # oof_predict already sorts; the holdout path returns raw order
        scores[name], imps[name] = score, imp.sort_values(ascending=False)
        m = guard_metrics(score, te_df, thr)
        m["features"], m["thr"] = name, thr
        m["rho"] = spearmanr(score, (te_df.grad_final_iou - te_df.headsel_iou)).statistic
        rows.append(m)

    r = pd.DataFrame(rows)
    print("=== guard on held-out images ===")
    print(f"{'features':<8} | {'thr':>7} | {'rho':>6} | {'keep%':>6} | {'grad':>7} | "
          f"{'GUARDED':>7} | {'vs grad':>8} | {'z':>5} | {'oracle':>7}")
    print("-" * 84)
    for _, x in r.iterrows():
        print(f"{x.features:<8} | {x.thr:>7.4f} | {x.rho:>6.3f} | {x.kept:>6.1%} | "
              f"{x.grad:>7.4f} | {x.guarded:>7.4f} | {x.vs_grad:>+8.4f} | "
              f"{x.z:>5.1f} | {x.oracle:>7.4f}")

    # how the guard changes the per-group story -- the strong group is where
    # the ungated gradient loses, so that is where the guard has to earn its keep
    best = "all" if "all" in scores else r.iloc[0].features
    thr_best = float(r[r.features == best].thr.iloc[0])
    grp = within_image_groups(te_df)
    print(f"\n=== metric change by within-image group (features '{best}', "
          f"thr {thr_best:+.4f}) ===")
    print(f"{'group':>7} | {'n':>5} | {'headsel':>7} | {'grad':>7} | {'GUARDED':>7} | "
          f"{'guard-grad':>10} | {'z':>5} | {'oracle':>7}")
    print("-" * 76)
    for name in ("all",) + GROUP_NAMES:
        sel = np.ones(len(te_df), bool) if name == "all" else (grp == name)
        if not sel.any():
            continue
        m = guard_metrics(scores[best][sel], te_df[sel], thr_best)
        print(f"{name:>7} | {m['n']:>5} | {m['headsel']:>7.4f} | {m['grad']:>7.4f} | "
              f"{m['guarded']:>7.4f} | {m['vs_grad']:>+10.4f} | {m['z']:>5.1f} | "
              f"{m['oracle']:>7.4f}")

    print(f"\ntop features ('{best}'):")
    print(imps[best].head(15).to_string())
    tot = imps[best].sum()
    for tag in ("t1_", "t2_"):
        print(f"{tag}* share of importance: {imps[best].filter(like=tag).sum() / tot:.1%}")

    if a.out_csv:
        Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
        r.to_csv(a.out_csv, index=False)
        imps[best].rename("importance").to_csv(
            a.out_csv.replace(".csv", "_importance.csv"))
        print(f"\nSaved -> {a.out_csv}")


if __name__ == "__main__":
    main()
