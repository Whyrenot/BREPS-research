"""
export_selector_tables.py
=========================
The two tables the SAM3 selector study was reported in, as CSV, from a finished
`train_selectors.py` run -- so the same study on another backbone is a diff
rather than a retelling.

    TABLE 1  out-of-fold, all 500 images, per task x group:
             undefended / argmax f_pred (today's defence) / CatBoost / logistic
             / oracle, each as an absolute IoU and as a relative gain over
             undefended, with image-clustered z. T1 also carries the genuine
             early-stop row and its forward count.

    TABLE 2  the within-image weak<->strong gap on ONE held-out fold, and its
             decomposition: a policy can close the gap by lifting weak or by
             breaking strong, and only the decomposition tells them apart.

WHY IT EXISTS. train_selectors.py writes metrics.csv -- every policy it knows,
against the task's own incumbent. These two tables re-reference everything to
the UNDEFENDED mask (the only baseline that means the same thing in all three
tasks, and the only one comparable across backbones) and add the gap analysis,
which is not a per-case mean and so cannot be recovered from metrics.csv at all.
Both are recomputed here from `t*_scores.csv` + the candidate pool, by the same
code paths train_selectors used (selector_data for the groups,
selector_policies for the grid and the clustered z), so no number depends on
this script agreeing with a table someone typed out.

NOTHING IS REFIT. The scores CSVs already hold every head's out-of-fold
prediction, so this is a laptop job: reshape, argmax, group.

    python scripts/export_selector_tables.py --tag sam3 \\
        --pool results/sam3_selector_pool_m.csv \\
        --run_dir results/selectors --out_dir results/selector_tables

    # then, once two or more models are in out_dir:
    python scripts/export_selector_tables.py --merge --out_dir results/selector_tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import selector_data as SD
import selector_policies as SP

# Which head answers "CatBoost" and "logistic" for each task, and which column
# reproduces today's defence. These are the choices the SAM3 write-up made:
#   t1  the ascent's checkpoints   -- today = argmax of f_pred over them
#   t2  16 bon draws + undefended  -- today = the bon pool's own argmax, which
#                                     is `method_slot == "bon"`, NOT an argmax
#                                     that is also allowed to keep undefended
#   t3  t2's pool + grad_final     -- today = argmax f_pred over all 18
# cb/lr are the heads that won on OOF for that task; --cb_head / --lr_head
# override them without editing this table.
TASK_SPEC = {
    "t1": {"fpred": "f_pred", "cb": "cb_rank", "lr": "lr_ens"},
    "t2": {"fpred": "_bonpick", "cb": "cb_ens", "lr": "lr_ens"},
    "t3": {"fpred": "f_pred", "cb": "cb_rank", "lr": "lr_ens"},
}
TASK_LABEL = {"t1": "1. early stop", "t2": "2. BoN + initial",
              "t3": "3. BoN + initial + gradient"}
# pool columns the grid needs but the scores CSV does not carry
POOL_COLS = ["image_name", "kind", "user", "attempt", "cand_kind", "cand_id",
             "method_slot", "step", "f_pred", "n_batched_fwd", "gt_undef_iou",
             "target_iou"]
KEY = ["case", "cand_kind", "cand_id", "step"]


def load_scored(pool: pd.DataFrame, run_dir: Path, task: str) -> pd.DataFrame:
    """The task's scored candidates, in the row order train_selectors wrote --
    which is the order CaseGrid reshapes, so it must not be disturbed."""
    path = run_dir / f"{task}_scores.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run train_selectors.py --tasks {task} first")
    sc = pd.read_csv(path)
    sc["_order"] = np.arange(len(sc))
    keep = ["case"] + [c for c in POOL_COLS if c in pool.columns]
    drop = [c for c in keep if c in sc.columns and c != "case" and c not in KEY]
    m = sc.drop(columns=drop).merge(pool[keep], on=KEY, how="left",
                                    validate="one_to_one")
    if len(m) != len(sc) or m.image_name.isna().any():
        raise SystemExit(f"{task}: {path.name} and the pool do not line up -- "
                         "is --pool the CSV this run was trained on?")
    m = m.sort_values("_order", kind="mergesort").reset_index(drop=True)
    # the fixed-policy indicator column train_selectors builds on the fly
    m["_bonpick"] = (m.method_slot == "bon").astype(float)
    return m


def policy_ious(d: pd.DataFrame, task: str, spec: dict, stop_thr: float):
    """Per-case chosen IoU for every policy in the two tables.

    Returns (frame indexed by case, group label per case, image per case).
    """
    grid = SP.CaseGrid(d)
    out = {}
    for name, col in (("f_pred", spec["fpred"]), ("catboost", spec["cb"]),
                      ("logistic", spec["lr"])):
        if col not in d.columns:
            raise SystemExit(f"{task}: no column {col!r} in the scores CSV")
        out[name] = grid.argmax(grid.col(d, col))[0]
    out["oracle"] = grid.oracle()[0]
    cost = {k: float(grid.cost[:, -1].mean()) for k in out}

    if task == "t1" and "cb_cont" in d.columns and not d.cb_cont.isna().all():
        # genuine early stopping: halt the first time the "will any later
        # checkpoint beat my best so far" head drops below stop_thr, and keep
        # the best checkpoint seen. The quality score is cb_ens, as in
        # train_selectors -- the stop head says when, not which.
        q = grid.col(d, "cb_ens" if "cb_ens" in d.columns else spec["cb"])
        p = 1.0 / (1.0 + np.exp(-grid.col(d, "cb_cont")))
        iou, c, _ = grid.stop_prob(q, p, stop_thr)
        out["stop"] = iou
        cost["stop"] = float(c.mean())

    frame = pd.DataFrame(out, index=pd.Index(grid.cases, name="case"))
    # undefended is a property of the case, not of this pool: T1's candidates
    # are gradient checkpoints and contain no undefended mask at all, so it is
    # read off gt_undef_iou -- identical for all three tasks by construction.
    frame.insert(0, "undef", d.groupby("case", sort=False).gt_undef_iou.first()
                 .reindex(frame.index).values)
    grp = SD.image_groups(d).reindex(frame.index).values
    img = pd.Series(frame.index).str.split("|").str[0].values
    frame.attrs["cost"] = cost
    frame.attrs["fold"] = grid.col(d, "_fold")[:, 0]
    return frame, grp, img


def table1(frame: pd.DataFrame, grp, img, task: str, tag: str) -> pd.DataFrame:
    """One row per group: every policy against the undefended mask."""
    pols = [c for c in frame.columns if c != "undef"]
    rows = []
    for g in SP.GROUPS:
        m = np.ones(len(frame), bool) if g == "all" else (grp == g)
        if not m.any():
            continue
        sub, im = frame[m], img[m]
        u = sub.undef.to_numpy()
        r = {"model": tag, "task": task, "task_label": TASK_LABEL[task],
             "group": g, "n": int(m.sum()), "undef": float(u.mean())}
        for p in pols:
            v = sub[p].to_numpy()
            dv, z = SP.image_clustered_z(v - u, im)
            r[p] = float(v.mean())
            r[f"d_{p}_vs_undef"] = dv
            r[f"pct_{p}_vs_undef"] = 100.0 * dv / u.mean() if u.mean() else np.nan
            r[f"z_{p}_vs_undef"] = z
            if p != "f_pred":
                dd, zz = SP.image_clustered_z(v - sub.f_pred.to_numpy(), im)
                r[f"d_{p}_vs_fpred"] = dd
                r[f"z_{p}_vs_fpred"] = zz
        for p, c in frame.attrs["cost"].items():
            r[f"fwd_{p}"] = c
        rows.append(r)
    return pd.DataFrame(rows)


def table2(frame: pd.DataFrame, grp, img, task: str, tag: str, fold: int
           ) -> pd.DataFrame:
    """The weak<->strong gap INSIDE each image, on one held-out fold.

    Per image: mean over its weak annotations minus mean over its strong ones.
    Averaged over images, and every delta paired per image before its standard
    error -- one image is one observation, which is what makes n the number of
    images and not the number of annotations.
    """
    sel = frame.attrs["fold"] == fold
    if not sel.any():
        raise SystemExit(f"fold {fold} is empty -- --gap_fold out of range?")
    f, g, i = frame[sel], grp[sel], img[sel]
    pols = ["undef"] + [c for c in f.columns if c != "undef"]

    def per_image(col):
        """(weak mean, strong mean, gap) per image, images as the index."""
        v = pd.DataFrame({"img": i, "grp": g, "v": f[col].to_numpy()})
        w = v[v.grp == "weak"].groupby("img").v.mean()
        s = v[v.grp == "strong"].groupby("img").v.mean()
        both = w.index.intersection(s.index)
        return w[both], s[both], (s[both] - w[both])

    w0, s0, gap0 = per_image("undef")
    rows = []
    for p in pols:
        w, s, gap = per_image(p)
        r = {"model": tag, "task": task, "task_label": TASK_LABEL[task],
             "policy": p, "n_images": int(len(gap)),
             "weak": float(w.mean()), "strong": float(s.mean()),
             "gap": float(gap.mean()),
             "gap_pct_of_strong": (100.0 * gap.mean() / s.mean()
                                   if s.mean() else np.nan)}
        if p == "undef":
            r.update({k: np.nan for k in
                      ("d_gap", "z_gap", "pct_gap_reduction", "d_weak", "z_weak",
                       "pct_weak", "d_strong", "z_strong", "pct_strong")})
        else:
            ix = gap.index.to_numpy()
            dg, zg = SP.image_clustered_z((gap - gap0[gap.index]).to_numpy(), ix)
            dw, zw = SP.image_clustered_z((w - w0[w.index]).to_numpy(), ix)
            ds, zs = SP.image_clustered_z((s - s0[s.index]).to_numpy(), ix)
            r.update({
                "d_gap": dg, "z_gap": zg,
                "pct_gap_reduction": 100.0 * dg / gap0.mean(),
                "d_weak": dw, "z_weak": zw,
                "pct_weak": 100.0 * dw / w0.mean(),
                "d_strong": ds, "z_strong": zs,
                "pct_strong": 100.0 * ds / s0.mean(),
            })
        rows.append(r)
    return pd.DataFrame(rows)


def merge(out_dir: Path) -> None:
    """Stack every model's tables into one file each, models as a column."""
    for which in ("tab1_oof", "tab2_gap"):
        parts = sorted(p for p in out_dir.glob(f"*_{which}.csv")
                       if not p.name.startswith("compare_"))
        if not parts:
            print(f"[skip] no *_{which}.csv in {out_dir}")
            continue
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        dst = out_dir / f"compare_{which}.csv"
        df.to_csv(dst, index=False)
        print(f"Saved {dst}  ({df.model.nunique()} models: "
              f"{', '.join(sorted(df.model.unique()))})")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", help="the candidate CSV from dump_selector_data.py")
    p.add_argument("--run_dir", help="train_selectors.py --out_dir for that pool")
    p.add_argument("--out_dir", default="results/selector_tables")
    p.add_argument("--tag", help="model label that goes in the `model` column")
    p.add_argument("--tasks", nargs="+", default=["t1", "t2", "t3"])
    p.add_argument("--gap_fold", type=int, default=0,
                   help="fold whose test images table 2 is computed on")
    p.add_argument("--stop_threshold", type=float, default=0.2,
                   help="continue-probability below which T1's stop rule halts")
    p.add_argument("--cb_head", help="override the CatBoost head for every task")
    p.add_argument("--lr_head", help="override the linear head for every task")
    p.add_argument("--merge", action="store_true",
                   help="only restack the per-model tables already in --out_dir")
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        merge(out)
        return
    for need in ("pool", "run_dir", "tag"):
        if not getattr(a, need):
            raise SystemExit(f"--{need} is required (or pass --merge)")

    pool = SD.load_pool(a.pool)
    print(f"{a.tag}: {len(pool)} rows | {pool.case.nunique()} cases | "
          f"{pool.image_name.nunique()} images", flush=True)
    t1s, t2s = [], []
    for task in a.tasks:
        spec = dict(TASK_SPEC[task])
        if a.cb_head:
            spec["cb"] = a.cb_head
        if a.lr_head:
            spec["lr"] = a.lr_head
        d = load_scored(pool, Path(a.run_dir), task)
        frame, grp, img = policy_ious(d, task, spec, a.stop_threshold)
        t1s.append(table1(frame, grp, img, task, a.tag))
        t2s.append(table2(frame, grp, img, task, a.tag, a.gap_fold))
        print(f"  {task}: {len(frame)} cases, heads "
              f"cb={spec['cb']} lr={spec['lr']} vs {spec['fpred']}", flush=True)

    for which, parts in (("tab1_oof", t1s), ("tab2_gap", t2s)):
        dst = out / f"{a.tag}_{which}.csv"
        pd.concat(parts, ignore_index=True).to_csv(dst, index=False)
        print(f"Saved {dst}")
    merge(out)


if __name__ == "__main__":
    main()
