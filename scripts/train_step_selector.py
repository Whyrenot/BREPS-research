"""
train_step_selector.py
======================
Train and score a CatBoost model that picks WHICH GRADIENT STEP produces the
mask, from the per-candidate dump written by
`refine_box_iou_grad.py --dump_candidates`.

Why this is worth doing. Over a 500-image SAM3 run the ascent's own trajectory
oracle (`grad_best`) beat its final step by +1.83% -- comparable to the entire
effect of the defence, and larger than anything a grad-vs-headsel guard
achieved. The ascent does not diverge; it overshoots. The predicted-IoU head
cannot find the turning point on its own: it rises monotonically while the true
IoU peaks and falls, which is exactly why rho(d_pred, d_true) is only 0.165.

Two policies are reported, because they answer different questions:

  select  -- run every step, then keep the checkpoint with the highest
             PREDICTED quality. Costs the same as a full ascent; strictly the
             better policy for quality, and the honest ceiling for "can a model
             read the trajectory at all".
  stop    -- genuine early stopping: walk the checkpoints in order and stop the
             first time the predicted quality falls below the running best by
             more than --patience. Saves compute, and can only do worse than
             `select`, since it may halt before a later peak.

Baselines it is scored against: the first checkpoint (head-select, the ascent
never happened), the last (`grad_final`, today's behaviour), picking by the
model's own predicted IoU (`c_pred`, the naive selector), and the oracle over
checkpoints.

Honesty constraints, same as train_gate.py:
  * Validation is GroupKFold on image_name -- the 25 annotations of one image
    share an object, so a random split leaks.
  * Features are whitelisted to the causal `c_*` block plus step position.
    `target_iou` is the label; nothing derived from the ground truth is a
    feature.

    python scripts/train_step_selector.py --csv results/sam3_steps_full.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

CASE_KEYS = ["image_name", "kind", "user", "attempt"]
DROP = set(CASE_KEYS) | {"target_iou", "start"}


def load(paths) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for k in CASE_KEYS:
        if k not in df.columns:
            df[k] = ""
    df[CASE_KEYS] = df[CASE_KEYS].fillna("")
    if "target_iou" not in df.columns:
        raise SystemExit("no target_iou column -- the dump was written without GT")
    df["case"] = df[CASE_KEYS].astype(str).agg("|".join, axis=1)
    return df.sort_values(["case", "start", "step"], kind="mergesort").reset_index(drop=True)


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Causal columns only, minus anything constant or entirely missing.

    With --multistart 1 several agreement columns degenerate (there is no other
    candidate to agree with, and the cluster is trivially everything), so they
    are dropped rather than fed as noise.
    """
    cols = [c for c in df.columns
            if (c.startswith("c_") or c in ("step", "step_frac", "is_start0"))
            and c not in DROP]
    keep = []
    for c in cols:
        v = df[c]
        if v.notna().sum() == 0:
            continue
        if v.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep


def oof_scores(df, cols, groups, folds, seed, iters, depth, lr):
    from catboost import CatBoostRegressor
    oof = np.zeros(len(df))
    imps = []
    folds = min(folds, groups.nunique())
    for tr, te in GroupKFold(n_splits=folds).split(df, df.target_iou, groups):
        m = CatBoostRegressor(iterations=iters, depth=depth, learning_rate=lr,
                              loss_function="RMSE", random_seed=seed, verbose=0)
        m.fit(df.iloc[tr][cols], df.iloc[tr].target_iou)
        oof[te] = m.predict(df.iloc[te][cols])
        imps.append(pd.Series(m.get_feature_importance(), index=cols))
    return oof, pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


def pick_argmax(g: pd.DataFrame, score: str) -> pd.Series:
    """Highest scoring checkpoint of a case."""
    return g.loc[g[score].idxmax()]


def pick_stop(g: pd.DataFrame, score: str, patience: float) -> pd.Series:
    """Walk checkpoints in step order; stop once the score has fallen more than
    `patience` below the running best, and return the best seen so far. This is
    causal -- it never looks at a checkpoint it would not have computed."""
    best_i, best_v = g.index[0], g[score].iloc[0]
    for i, v in zip(g.index, g[score]):
        if v > best_v:
            best_i, best_v = i, v
        elif best_v - v > patience:
            break
    return g.loc[best_i]


def report(rows: dict, df: pd.DataFrame, first: pd.Series, last: pd.Series):
    """Mean IoU of each policy plus a paired, image-clustered z against the
    current behaviour (the last checkpoint)."""
    img = df.groupby("case").image_name.first()
    out = []
    for name, v in rows.items():
        d = pd.Series(v.values - last.values, index=v.index)
        per = d.groupby(img).mean()
        se = per.std() / np.sqrt(max(1, per.size))
        out.append({"policy": name, "iou": v.mean(), "vs_last": d.mean(),
                    "z": abs(d.mean()) / se if se > 0 else np.nan})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True,
                   help="candidate dump(s) from --dump_candidates; globs allowed")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--patience", type=float, default=0.005,
                   help="early-stop slack: halt once the predicted quality has "
                        "fallen this far below the running best")
    p.add_argument("--out_csv", default=None)
    a = p.parse_args()

    paths = [q for pat in a.csv for q in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no CSVs matched {a.csv}")
    df = load(paths)
    if df.start.nunique() > 1:
        print(f"[note] dump has {df.start.nunique()} starts; this script scores "
              f"STEP selection, so all starts are pooled as candidates")

    cols = feature_cols(df)
    n_ck = df.groupby("case").size()
    print(f"{len(df)} candidate rows / {df.case.nunique()} cases / "
          f"{df.image_name.nunique()} images")
    print(f"checkpoints per case: {n_ck.min()}..{n_ck.max()} | features: {len(cols)}")
    dropped = [c for c in df.columns
               if (c.startswith("c_") and c not in cols and c not in DROP)]
    if dropped:
        print(f"dropped as constant/empty: {', '.join(dropped)}")

    oof, imp = oof_scores(df, cols, df.image_name, a.folds, a.seed,
                          a.iterations, a.depth, a.lr)
    df["pred_q"] = oof

    grp = df.groupby("case", sort=False)
    first = grp.apply(lambda g: g.target_iou.iloc[0], include_groups=False)
    last = grp.apply(lambda g: g.target_iou.iloc[-1], include_groups=False)
    policies = {
        "first checkpoint (= head-select)": first,
        "last checkpoint (= grad_final, today)": last,
        "argmax c_pred (naive)": grp.apply(
            lambda g: pick_argmax(g, "c_pred").target_iou, include_groups=False),
        "SELECT by CatBoost": grp.apply(
            lambda g: pick_argmax(g, "pred_q").target_iou, include_groups=False),
        f"STOP by CatBoost (patience {a.patience})": grp.apply(
            lambda g: pick_stop(g, "pred_q", a.patience).target_iou,
            include_groups=False),
        "ORACLE over checkpoints": grp.target_iou.max(),
    }
    r = report(policies, df, first, last)

    print(f"\n{'policy':<40} | {'IoU':>7} | {'vs last':>17}")
    print("-" * 72)
    for _, x in r.iterrows():
        tail = "        -" if x.policy.startswith("last") else f"{x.vs_last:+.4f} (z={x.z:>4.1f})"
        print(f"{x.policy:<40} | {x.iou:>7.4f} | {tail:>17}")

    ceil = policies["ORACLE over checkpoints"].mean() - last.mean()
    got = policies["SELECT by CatBoost"].mean() - last.mean()
    if ceil > 0:
        print(f"\nSELECT captures {got / ceil:.1%} of the checkpoint oracle "
              f"(+{ceil:.4f} available)")

    # which step each policy lands on -- a selector that always picks the last
    # step has learned nothing, and this is how you see that at a glance
    sel_step = grp.apply(lambda g: pick_argmax(g, "pred_q").step, include_groups=False)
    print(f"\nstep chosen by SELECT: median {sel_step.median():.0f}, "
          f"mean {sel_step.mean():.1f}, last-step share {(sel_step == df.step.max()).mean():.1%}")

    # within-image groups, ranked by the first checkpoint -- the same 20/60/20
    # stratification used everywhere else in this project
    img = df.groupby("case").image_name.first()
    g = pd.Series(index=first.index, dtype=object)
    for _, sub in first.groupby(img):
        o = sub.sort_values(kind="mergesort").index
        n = len(o); lo, hi = int(round(.2 * n)), int(round(.8 * n))
        for nm, ix in (("weak", o[:lo]), ("mid", o[lo:hi]), ("strong", o[hi:])):
            g[ix] = nm
    print(f"\n{'group':>7} | {'n':>5} | {'first':>7} {'last':>7} {'SELECT':>7} "
          f"{'STOP':>7} {'ORACLE':>7} | {'SELECT-last':>12}")
    print("-" * 78)
    for nm in ("weak", "mid", "strong"):
        m = (g == nm).values
        if not m.any():
            continue
        sel = policies["SELECT by CatBoost"][m]
        d = sel.values - last[m].values
        per = pd.Series(d).groupby(img[m].values).mean()
        se = per.std() / np.sqrt(max(1, per.size))
        print(f"{nm:>7} | {m.sum():>5} | {first[m].mean():>7.4f} {last[m].mean():>7.4f} "
              f"{sel.mean():>7.4f} "
              f"{policies[f'STOP by CatBoost (patience {a.patience})'][m].mean():>7.4f} "
              f"{policies['ORACLE over checkpoints'][m].mean():>7.4f} | "
              f"{d.mean():>+7.4f} (z={abs(d.mean()) / se if se > 0 else 0:>3.1f})")

    print(f"\ntop features:")
    print(imp.head(12).to_string())

    if a.out_csv:
        Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
        r.to_csv(a.out_csv, index=False)
        imp.rename("importance").to_csv(a.out_csv.replace(".csv", "_importance.csv"))
        print(f"\nSaved -> {a.out_csv}")


if __name__ == "__main__":
    main()
