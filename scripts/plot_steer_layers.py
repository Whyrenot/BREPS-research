"""
plot_steer_layers.py
====================
The two-panel layer figure of the critical-shift study (results/00_critical_shifts
layer_probe.png + steer_layers_*.png, as one figure) for a steer_models_scan.py
run on the user study. CPU only, from the run's dump; no model.

  a) probe     per decoder site and token group: ROC AUC of an L2 logistic
               regression (standardised, balanced classes) that tells the
               bottom50 annotations of an image from its top50, out of fold on
               COCO A (states_A.npz; the scan's image folds). Only A: the scan
               keeps no test-set states.
  b) steering  per site and method (DM, the random direction of DM's norm):
               the share of the gap to the image's best annotation that is
               closed on bottom50, sum d / sum (IoU_best - IoU_base), at a harm
               budget -- the token group and alpha with the best share among
               those whose mean IoU change on top50 is >= -budget, chosen on
               COCO A out of fold, frozen, scored on --datasets (equal weights).
               The critical-shift figure bounded the pixel change of good masks;
               the scan stores IoUs only, so the budget is on top50's IoU.

Outputs: <out>.png and <out>.csv (the plotted values, per dataset too).

    python scripts/plot_steer_layers.py --run_dir exp_res/steer_scan_SAM2.1 --mode 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN_LINES = [("all", "Весь узел", "#2a78d6", "o"), ("mask0", "Токен маски 0", "#eb6834", "s"),
               ("mask123", "Токены масок 1–3", "#1baf7a", "D"), ("prompt", "Токены подсказки", "#eda100", "^"),
               ("iou", "IoU-токен", "#e87ba4", "v"), ("obj", "Obj-токен", "#6250d6", "P")]
METHOD_LINES = [("dm", "Разность средних (DM)", "#2a78d6", "o"), ("rand", "Случайный вектор той же нормы", "#1baf7a", "D")]
MODEL_RU = {"SAM": "SAM ViT-B", "SAM2.1": "SAM 2.1 Hiera-B+", "SAM3": "SAM 3"}


def group_rows(layout: dict, site_kind: str, g: str):
    """Token rows of a probe group at a site, None for the whole tensor; [] if absent."""
    if site_kind != "tok":
        return None if g == "all" else []
    if g == "all":
        return None
    if g == "iou":
        return [1 if layout["obj_score_token"] else 0]
    if g == "obj":
        return [0] if layout["obj_score_token"] else []
    return layout["rows"][g]


def probe_panel(run: Path, meta: dict, A: pd.DataFrame, C: float) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    st = np.load(run / "states_A.npz")
    y = (A.group.to_numpy() == "bottom50").astype(int)
    fold = A.fold.to_numpy()
    rows = []
    for s in meta["sites"]:
        X0 = st[s["id"]].astype(np.float64)
        if len(X0) != len(A):
            raise SystemExit(f"states_A {s['id']}: {len(X0)} rows, per_case A: {len(A)}")
        for g, *_ in TOKEN_LINES:
            r = group_rows(meta["layout"], s["kind"], g)
            if r is not None and not r:
                continue
            X = (X0 if r is None else X0[:, r]).reshape(len(X0), -1)
            score = np.full(len(X), np.nan)
            for k in np.unique(fold):
                tr, te = fold != k, fold == k
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
                score[te] = clf.fit(X[tr], y[tr]).decision_function(X[te])
            rows.append({"panel": "a", "site": s["id"], "line": g, "value": roc_auc_score(y, score), "dataset": "COCO_A"})
    return pd.DataFrame(rows)


def restore(df: pd.DataFrame, col: str) -> float:
    b = df[df.group == "bottom50"]
    gap = (b.best_iou - b.base_iou).sum()
    return float((b[col] - b.base_iou).sum() / gap) if gap > 0 else np.nan


def steer_panel(meta: dict, cfg: pd.DataFrame, per: pd.DataFrame, datasets, budget: float) -> pd.DataFrame:
    A = per[per.key == "COCO_A"]
    t50 = A[A.group == "top50"]
    rows = []
    for s in meta["sites"]:
        for m, *_ in METHOD_LINES:
            cand = cfg[(cfg.site == s["id"]) & (cfg.method == m)]
            best, best_v = None, -np.inf
            for c in cand.itertuples():
                col = f"iou_c{c.cid}"
                if (t50[col] - t50.base_iou).mean() < -budget:
                    continue
                v = restore(A, col)
                if v > best_v + 1e-12:
                    best, best_v = c, v
            vals = {}
            for d in datasets:
                vals[d] = restore(per[per.key == d], f"iou_c{best.cid}") if best is not None else np.nan
            rows.append({"panel": "b", "site": s["id"], "line": m, "value": float(np.nanmean(list(vals.values())))
                         if best is not None else np.nan, "dataset": "+".join(datasets),
                         "group": best.group if best is not None else "", "alpha": best.alpha if best is not None else np.nan,
                         "restore_A": best_v if best is not None else np.nan, **{f"restore_{d}": v for d, v in vals.items()}})
    return pd.DataFrame(rows)


def plot(meta: dict, a: pd.DataFrame, b: pd.DataFrame, out: Path, budget: float, datasets, mode: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ink, ink2, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.titlelocation": "left", "axes.titlesize": 12.5})
    sites = [s["id"] for s in meta["sites"]]
    x = np.arange(len(sites))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 8.0), sharex=True, gridspec_kw={"hspace": 0.12})
    hl = sites.index("L2.norm2") if "L2.norm2" in sites else None
    for ax, nd in ((ax1, 2), (ax2, 1)):
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, nd=nd: f"{v:.{nd}f}".replace(".", ",")))
        if hl is not None:
            ax.axvspan(hl - 0.4, hl + 0.4, color="#e9f1fc", zorder=0)
    for g, label, col, mk in TOKEN_LINES:
        sub = a[a.line == g].set_index("site").reindex(sites)
        have = sub.value.notna().to_numpy()
        if have.any():
            # a token group exists only on token-stream sites: join its own points across the others
            ax1.plot(x[have], sub.value.to_numpy()[have], color=col, marker=mk, ms=7, lw=2.6 if g == "all" else 1.8,
                     label=label, mec="white", mew=1.2, zorder=3 if g == "all" else 2)
    ax1.axhline(0.5, color="#898781", lw=1.2, ls=":")
    ax1.set_ylabel("ROC AUC зонда:\nbottom50 против top50")
    ax1.set_title("а) Различимость слабых разметок по активациям узла (COCO A, вне фолда)", color=ink)
    ax1.legend(ncol=3, frameon=False, fontsize=10, loc="upper left")
    for m, label, col, mk in METHOD_LINES:
        sub = b[b.line == m].set_index("site").reindex(sites)
        ax2.plot(x, sub.value, color=col, marker=mk, ms=7, lw=2.6 if m == "dm" else 1.8, label=label,
                 mec="white", mew=1.2, zorder=3 if m == "dm" else 2)
    ax2.axhline(0, color=axis, lw=1.2)
    ax2.set_ylabel("Доля закрытого разрыва\nдо лучшей разметки (bottom50)")
    ds = ", ".join(d.replace("_", " ") for d in datasets)
    ax2.set_title(f"б) Steering в узле: падение IoU top50 не более {budget:g}".replace(".", ",") + f"  ·  {ds}",
                  color=ink)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sites, rotation=35, ha="right")
    out_mode = "multimask=True, база — head-select" if mode else "multimask=False, база — один выход"
    ax2.set_xlabel(f"Узел декодера масок {MODEL_RU.get(meta['model_name'], meta['model_name'])} "
                   f"(от входа к выходу) · {out_mode}")
    ax2.legend(ncol=2, frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0: multimask=False, 1: multimask=True")
    p.add_argument("--datasets", nargs="*", default=["COCO_B", "Berkeley", "GrabCut"], help="panel b, frozen choice")
    p.add_argument("--budget", type=float, default=0.005, help="max mean IoU drop on top50 (COCO A, out of fold)")
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the probes")
    p.add_argument("--out", default="", help="default: <run_dir>/layers_mm<mode>")
    args = p.parse_args(argv)
    run = Path(args.run_dir)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    cfg = pd.read_csv(run / "configs.csv")
    cols = ["dataset", "half", "fold", "group", "base_iou", "best_iou"] + [f"iou_c{c}" for c in cfg.cid]
    per = pd.read_csv(run / f"per_case_mm{args.mode}.csv.gz", usecols=cols, dtype={"half": str})
    per["key"] = np.where(per.dataset == "COCO", "COCO_" + per.half, per.dataset)
    a = probe_panel(run, meta, per[per.key == "COCO_A"].reset_index(drop=True), args.C)
    b = steer_panel(meta, cfg, per, args.datasets, args.budget)
    out = Path(args.out) if args.out else run / f"layers_mm{args.mode}"
    pd.concat([a, b], ignore_index=True).to_csv(out.with_suffix(".csv"), index=False)
    plot(meta, a, b, out, args.budget, args.datasets, args.mode)
    print(f"wrote {out.with_suffix('.png')} and .csv")
    print(b[["site", "line", "group", "alpha", "restore_A", "value"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
