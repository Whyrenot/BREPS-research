"""
selector_steer.py
=================
What does the CatBoost selector gain when steering masks join its pool? Runs
locally on CPU: the server's pool (dump_selector_data.py: undefended + 16
best_of_n + the gradient checkpoints) plus the steering candidates of
dump_steer_candidates.py, with the selector of transfer_selectors.py's "in"
regime -- the same feature builder (selector_data._build_choice), the same 5
image-level folds stratified by subset (train_selectors.make_folds), the same
YetiRank head and hyperparameters -- so every case is scored out of fold and
the only change between the tasks is which candidates the pool holds.

TASKS
  t4    undef + bon + grad            the reported pool (its cb_rank should
                                      reproduce tabA_transfer_cb_rank's sam1 t4)
  t4s   t4 + steer                    what steering adds on top of everything
  t2s   undef + bon + steer           no gradient ascent (x_iou_hsel / grad dropped, as T2)
  ts    undef + steer                 steering alone (x_iou_bon dropped too)
  Steering rows get c_is_steer and f_steer_alpha; their x_iou_grad is NaN (the
  ascent was not re-run for them).

REPORT (per subset, the micro mean over all cases, natural / medical apart;
groups bottom50 / top50 inside each image): undefended, best_of_n pick, grad
final, argmax f_pred of the t4 pool, CatBoost per task, the t4s oracle; IoU
and % to undefended, and the delta of each CatBoost task to t4's CatBoost with
a 95% bootstrap CI over images.

CHECKS: every pool case has its steering rows; the steering rows' chk_* (this
dump's undefended / best_of_n pick / head-select IoUs) against the pool's own
-- they must describe the same cases.

    python scripts/selector_steer.py --pool results/sam1_selector_pool_m.csv \\
        --steer results/sam1_steer_candidates_m.csv --out_dir results/selector_steer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import selector_data as SD  # noqa: E402
import selector_models as SM  # noqa: E402
from train_selectors import make_folds  # noqa: E402

TASKS = OrderedDict([("t4", (("undef", "bon", "grad"), [])),
                     ("t4s", (("undef", "bon", "grad", "steer"), [])),
                     ("t2s", (("undef", "bon", "steer"), ["x_iou_hsel", "x_iou_grad"])),
                     ("ts", (("undef", "steer"), ["x_iou_hsel", "x_iou_grad", "x_iou_bon"]))])
TASK_LABEL = {"t4": "CatBoost: undef + BoN + grad (как раньше)", "t4s": "CatBoost: undef + BoN + grad + steering",
              "t2s": "CatBoost: undef + BoN + steering", "ts": "CatBoost: undef + steering"}
MEDICAL = ("ACDC", "BUID", "MedScribble")
GROUPS = ("all", "bottom50", "top50")


def load(pool_path: str, steer_path: str, limit_images: int):
    pool = SD.load_pool(pool_path)
    st = pd.read_csv(steer_path, dtype={k: str for k in SD.CASE_KEYS})
    st[SD.CASE_KEYS] = st[SD.CASE_KEYS].fillna("")
    st["case"] = st[SD.CASE_KEYS].astype(str).agg("|".join, axis=1)
    st["dataset"] = st.image_name.str.split("_").str[0]
    if limit_images > 0:
        keep = sorted(pool.image_name.unique())
        keep = set(np.random.default_rng(0).choice(keep, size=min(limit_images, len(keep)), replace=False))
        pool, st = pool[pool.image_name.isin(keep)], st[st.image_name.isin(keep)]
    return pool.reset_index(drop=True), st.reset_index(drop=True)


def check_alignment(pool: pd.DataFrame, st: pd.DataFrame) -> dict:
    pc, sc = set(pool.case), set(st.case)
    one = st.drop_duplicates("case").set_index("case")
    out = {"pool_cases": len(pc), "steer_cases": len(sc), "missing_in_steer": len(pc - sc),
           "extra_in_steer": len(sc - pc),
           "steer_rows_per_case": sorted(st.groupby("case").size().unique().tolist())}
    for col, sel in (("chk_undef_iou", pool.cand_kind == "undef"), ("chk_bon_pick_iou", pool.method_slot == "bon"),
                     ("chk_hsel_iou", pool.method_slot == "hsel")):
        ref = pool[sel].drop_duplicates("case").set_index("case").target_iou
        d = (one[col] - ref.reindex(one.index)).abs()
        out[col] = {"max_abs": float(d.max()), "share_over_1e-3": float((d > 1e-3).mean()),
                    "missing": int(d.isna().sum())}
    out["ok"] = (out["missing_in_steer"] == 0 and out["extra_in_steer"] == 0
                 and out["chk_undef_iou"]["share_over_1e-3"] <= 0.01)
    return out


def merge(pool: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """The steering rows take the pool's gt_* of their case, so groups and the
    undefended reference are the pool's; chk_* are dropped (never features)."""
    u = pool[pool.cand_kind == "undef"].drop_duplicates("case").set_index("case")
    st = st.drop(columns=[c for c in st.columns if c.startswith("chk_")])
    for c in ("gt_undef_iou", "gt_clean_iou"):
        st[c] = st.case.map(u[c])
    return pd.concat([pool, st], ignore_index=True)


def build(comb: pd.DataFrame, task: str):
    kinds, drop = TASKS[task]
    d, cols = SD._build_choice(comb[comb.cand_kind.isin(kinds)].copy(), drop)
    if "steer" in kinds:
        d["c_is_steer"] = (d.cand_kind == "steer").astype(float)
        cols = cols + ["c_is_steer"]
    return d, cols


def oof_scores(d: pd.DataFrame, cols, folds, a, cache: Path, task: str) -> np.ndarray:
    key = d[["case", "cand_kind", "cand_id"]].astype(str).agg("|".join, axis=1)
    f = cache / f"oof_{task}.csv.gz"
    if f.is_file() and not a.refit:
        c = pd.read_csv(f)
        if len(c) == len(d) and (c.key.to_numpy() == key.to_numpy()).all():
            print(f"[{task}] cached OOF scores {f}", flush=True)
            return c.score.to_numpy()
    s = np.full(len(d), np.nan)
    for k, te_imgs in enumerate(folds):
        te = d.image_name.isin(te_imgs).to_numpy()
        t0 = time.time()
        s[te] = SM.fit_predict_catboost(d[~te], d[te], cols, "rank", a.iterations, a.threads, a.seed,
                                        depth=a.depth, learning_rate=a.lr)
        print(f"[{task}] fold {k}: {te.sum()} rows, {len(cols)} features ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame({"key": key, "score": s}).to_csv(f, index=False)
    return s


def pick(d: pd.DataFrame, by: np.ndarray) -> pd.Series:
    """target_iou of the row with the highest `by` in every case (first on ties)."""
    t = d.assign(_by=by)
    idx = t.groupby("case", sort=False)._by.idxmax()
    return pd.Series(t.target_iou.loc[idx].to_numpy(), index=idx.index)


def boot_ci(d: np.ndarray, images: np.ndarray, n_boot: int, seed: int):
    codes, img = np.unique(images, return_inverse=True)
    S = np.bincount(img, weights=d, minlength=len(codes))
    N = np.bincount(img, minlength=len(codes)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(codes), size=(n_boot, len(codes)))
    return np.percentile(S[idx].sum(1) / N[idx].sum(1), [2.5, 97.5])


def report(per_case: pd.DataFrame, methods, a) -> pd.DataFrame:
    """per_case: one row per case with its dataset, image, group, undef and a
    column per method. Blocks: every subset, then all / natural / medical."""
    blocks = [(ds, per_case.dataset == ds) for ds in sorted(per_case.dataset.unique())]
    nat = ~per_case.dataset.isin(MEDICAL)
    blocks += [(f"все {per_case.dataset.nunique()} наборов", np.ones(len(per_case), bool)),
               (f"натуральные ({per_case[nat].dataset.nunique()})", nat.to_numpy()),
               (f"медицина ({per_case[~nat].dataset.nunique()})", (~nat).to_numpy())]
    rows = []
    for name, m in blocks:
        B = per_case[np.asarray(m)]
        if B.empty:
            continue
        for meth in methods:
            for g in GROUPS:
                s = B if g == "all" else B[B.group == g]
                x, u = s[meth].to_numpy(float), s.undef.to_numpy(float)
                r = {"block": name, "method": meth, "group": g, "n": len(s), "iou": x.mean(),
                     "pct": (x.mean() - u.mean()) / u.mean() * 100, "d_vs_t4": np.nan, "lo": np.nan, "hi": np.nan}
                if meth.startswith("cb_") and meth != "cb_t4" and "cb_t4" in s:
                    d = x - s.cb_t4.to_numpy(float)
                    lo, hi = boot_ci(d, s.image_name.to_numpy(), a.n_boot, a.seed)
                    r.update(d_vs_t4=d.mean(), lo=lo, hi=hi)
                rows.append(r)
    return pd.DataFrame(rows)


def md(R: pd.DataFrame, labels: dict, checks: dict, a) -> str:
    L = ["# CatBoost-селектор с кандидатами steering (SAM-1 vit_b, user study `m`)", "",
         "Кандидаты steering: вектор COCO A (DM top50 − bottom50, `L2.norm2`, все 7 токенов), α = 1…16, на "
         "картинках COCO A — вектор других фолдов A. Селектор — CatBoost YetiRank как в transfer_selectors "
         f"(режим «in»: {a.iterations} итераций, depth {a.depth}, lr {a.lr}), 5 фолдов по картинкам, "
         "стратификация по наборам, каждый бокс оценён вне фолда. Ячейка — IoU (прирост к undef, %); "
         "Δ к CatBoost t4 — разность выбранных IoU с 95% CI (бутстреп по картинкам). Группы — по undef IoU "
         "внутри картинки (12 / 13).", "",
         "| набор | метод | all | bottom50 | top50 | Δ all к CatBoost t4 [95% CI] |", "|---|---|---|---|---|---|"]
    for block, B in R.groupby("block", sort=False):
        for i, (meth, s) in enumerate(B.groupby("method", sort=False)):
            s = s.set_index("group")
            cells = [f"{s.loc[g, 'iou']:.4f}" if meth == "undef" else f"{s.loc[g, 'iou']:.4f} ({s.loc[g, 'pct']:+.2f}%)"
                     for g in GROUPS]
            r = s.loc["all"]
            ci = "—" if not np.isfinite(r.d_vs_t4) else f"{r.d_vs_t4:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]"
            L.append(f"| {('**' + block + '**') if i == 0 else ''} | {labels.get(meth, meth)} | "
                     + " | ".join(cells) + f" | {ci} |")
    al = checks.get("alignment", {})
    L += ["", "## Проверки", "",
          f"- случаев в пуле {al.get('pool_cases')}, со steering {al.get('steer_cases')}, без steering "
          f"{al.get('missing_in_steer')}; строк steering на случай {al.get('steer_rows_per_case')};"]
    for c in ("chk_undef_iou", "chk_bon_pick_iou", "chk_hsel_iou"):
        if c in al:
            L.append(f"- {c} против пула: max |Δ| {al[c]['max_abs']:.2e}, доля > 1e-3 {al[c]['share_over_1e-3']:.2%};")
    if "share_steer" in checks:
        L.append(f"- доля выбранных кандидатов steering: {checks['share_steer']};")
    return "\n".join(L) + "\n"


def run(a) -> int:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool, st = load(a.pool, a.steer, a.limit_images)
    checks = {"alignment": check_alignment(pool, st)}
    print(json.dumps(checks["alignment"], indent=1), flush=True)
    if not checks["alignment"]["ok"] and not a.force:
        (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        print("CHECK FAILED: the steering rows do not match the pool (see checks.json; --force to go on)",
              file=sys.stderr)
        return 1
    comb = merge(pool, st)
    folds = make_folds(pool, a.folds, a.seed)

    u = pool[pool.cand_kind == "undef"].drop_duplicates("case")
    per_case = u[["case", "image_name", "dataset", "gt_undef_iou"]].rename(columns={"gt_undef_iou": "undef"})
    per_case = per_case.set_index("case")
    per_case["group"] = SD.image_halves(pool.drop_duplicates("case")).reindex(per_case.index)
    for slot, col in (("bon", "bon_pick"), ("grad", "grad_final")):
        per_case[col] = pool[pool.method_slot == slot].drop_duplicates("case").set_index("case").target_iou
    labels = {"undef": "undef", "bon_pick": "BoN (argmax f_pred)", "grad_final": "grad",
              "fpred_t4": "argmax f_pred (пул t4)"}
    methods = ["undef", "bon_pick", "grad_final", "fpred_t4"]
    share = {}
    for task in a.tasks:
        d, cols = build(comb, task)
        if task == "t4":
            per_case["fpred_t4"] = pick(d, d.f_pred.to_numpy())
        s = oof_scores(d, cols, folds, a, out, task)
        per_case[f"cb_{task}"] = pick(d, s)
        methods.append(f"cb_{task}")
        labels[f"cb_{task}"] = TASK_LABEL[task]
        if "steer" in TASKS[task][0]:
            t = d.assign(_s=s)
            chosen = t.loc[t.groupby("case", sort=False)._s.idxmax()]
            share[task] = f"{(chosen.cand_kind == 'steer').mean():.1%}"
            if task == "t4s":
                per_case["oracle_t4s"] = d.groupby("case").target_iou.max()
    if "oracle_t4s" in per_case:
        methods.append("oracle_t4s")
        labels["oracle_t4s"] = "оракул (пул t4 + steering)"
    checks["share_steer"] = share
    per_case = per_case.reset_index()
    per_case.to_csv(out / "per_case.csv", index=False)
    R = report(per_case, methods, a)
    R.to_csv(out / "table.csv", index=False)
    text = md(R, labels, checks, a)
    (out / "summary.md").write_text(text, encoding="utf-8")
    (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print("\n" + text.encode(enc, "replace").decode(enc))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default="results/sam1_selector_pool_m.csv")
    p.add_argument("--steer", default="results/sam1_steer_candidates_m.csv")
    p.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--threads", type=int, default=0, help="0 = all cores")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--limit_images", type=int, default=0, help="random images, for a smoke run")
    p.add_argument("--refit", action="store_true", help="ignore cached OOF scores")
    p.add_argument("--force", action="store_true", help="go on when the alignment check fails")
    p.add_argument("--out_dir", default="results/selector_steer")
    a = p.parse_args(argv)
    if a.threads <= 0:
        import multiprocessing
        a.threads = multiprocessing.cpu_count()
    return a


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
