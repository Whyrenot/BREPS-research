"""
steer_models_report.py
======================
Tables and one self-contained HTML page for a steer_models_scan.py run: the
model's own decoder graph with the steering sites clickable, and per site the
effect of the frozen vector on the user-study boxes.

SELECTION (no test data involved)
  Per output mode and site, the token group and alpha of DM with the largest
  mean IoU gain over all boxes of COCO half A, every A image steered with the
  vector of the other folds (ties -> the smaller alpha). The random direction
  is scored at the same group and alpha (same norm); the oracle patches the
  site's whole tensor from the image's best box. The best site is the one with
  the largest A gain. Everything on COCO B and the other subsets is frozen.

PER SITE, DATASET, GROUP (bottom50 / top50 / all, inside each image by the
multimask=False IoU)
  mean IoU, gain over the mode's baseline (mm0: the single output, mm1:
  head-select) with a paired 95% bootstrap CI over images, % of the baseline,
  share of boxes up / down by more than --updown_eps, the share of the gap to
  the image's best box that is closed, and how often the output head changes.

Outputs (in --run_dir unless --out_dir): steer_scan_report.html, selected.csv,
table_sites.csv (chosen configs with CIs), grid.csv (every config, means),
summary.md.

    python scripts/steer_models_report.py --run_dir exp_res/steer_scan_SAM2.1 \\
        --reference_csv results/sam21full_m.csv
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["image_name", "kind", "user", "attempt"]
GROUPS = ("all", "bottom50", "top50")
GSHORT = {"all": "all", "bottom50": "b50", "top50": "t50"}
MODES = (0, 1)
WHOLE = {"sparse": "prompt", "tok": "all", "img": "box", "vec": "all"}
GROUP_ORDER = ("all", "mask0", "mask123", "prompt", "box")
# steer_user_study.py, first run (EXPERIMENTS.md section 1): SAM-1, L2.norm2,
# all tokens, alpha = 1, delta IoU over all boxes
EXPECT_SAM1 = {"COCO_B": 0.0078, "Berkeley": 0.0062, "GrabCut": 0.0051}
REF_COLS = {"headsel_iou": "head-select", "best_of_n_iou": "best-of-N", "grad_final_iou": "grad"}


def ds_key(df: pd.DataFrame) -> np.ndarray:
    return np.where(df.dataset == "COCO", "COCO_" + df.half.astype(str), df.dataset.astype(str))


def load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    dt = {"image_name": str, "kind": str, "user": str, "attempt": str, "half": str}
    per = {}
    for mm in MODES:
        df = pd.read_csv(run_dir / f"per_case_mm{mm}.csv.gz", dtype=dt)
        df[["kind", "user", "attempt"]] = df[["kind", "user", "attempt"]].fillna("")
        df["key"] = ds_key(df)
        per[mm] = df
    return {"dir": run_dir, "meta": meta,
            "checks": json.loads((run_dir / "checks.json").read_text(encoding="utf-8")),
            "graph": json.loads((run_dir / "graph.json").read_text(encoding="utf-8")),
            "cfg": pd.read_csv(run_dir / "configs.csv"), "probe": pd.read_csv(run_dir / "probe.csv"),
            "vstats": pd.read_csv(run_dir / "vector_stats.csv"), "per": per}


def group_mask(df: pd.DataFrame, g: str) -> np.ndarray:
    return np.ones(len(df), bool) if g == "all" else (df.group.to_numpy() == g)


def boot_ci(d: np.ndarray, img: np.ndarray, n_boot: int, seed: int):
    """95% CI of the mean of every column of d (boxes x k), resampling images
    with their boxes (steer_user_study.boot_delta, vectorised over columns)."""
    codes, inv = np.unique(img, return_inverse=True)
    if len(codes) < 2 or n_boot <= 0:
        nan = np.full(d.shape[1], np.nan)
        return nan, nan
    S = np.zeros((len(codes), d.shape[1]))
    np.add.at(S, inv, d)
    N = np.bincount(inv, minlength=len(codes)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(codes), size=(n_boot, len(codes)))
    est = S[idx].sum(1) / N[idx].sum(1)[:, None]
    lo, hi = np.percentile(est, [2.5, 97.5], axis=0)
    return lo, hi


def select(R: dict) -> dict:
    """Per mode and site: DM's group and alpha with the largest mean gain on
    COCO A (out of fold), the random direction there, the whole-tensor oracle."""
    cfg = R["cfg"]
    kinds = {s["id"]: s["kind"] for s in R["meta"]["sites"]}
    out = {}
    for mm in MODES:
        A = R["per"][mm][R["per"][mm].key == "COCO_A"]
        gain = A[[f"iou_c{c}" for c in cfg.cid]].to_numpy().mean(0) - A.base_iou.mean()
        sel = {}
        for s in R["meta"]["sites"]:
            sid = s["id"]
            cand = cfg[(cfg.site == sid) & (cfg.method == "dm")].copy()
            cand["gi"] = cand.group.map(GROUP_ORDER.index)
            cand = cand.sort_values(["alpha", "gi"], kind="mergesort")
            g = gain[cand.cid.to_numpy()]
            c = cand.iloc[int(np.argmax(g))]
            rnd = cfg[(cfg.site == sid) & (cfg.method == "rand") & (cfg.group == c.group)
                      & np.isclose(cfg.alpha, c.alpha)].cid.iloc[0]
            orc = cfg[(cfg.site == sid) & (cfg.method == "oracle") & (cfg.group == WHOLE[kinds[sid]])].cid.iloc[0]
            sel[sid] = {"group": c.group, "alpha": float(c.alpha), "dose": float(c.dose), "dm": int(c.cid),
                        "rand": int(rnd), "oracle": int(orc), "score_A": float(g.max())}
        best = max(sel, key=lambda k: sel[k]["score_A"])
        out[mm] = {"sites": sel, "best": best}
    return out


def fixed_cids(cfg: pd.DataFrame) -> dict | None:
    """L2.norm2, all tokens, alpha = 1: the configuration carried over from SAM-1."""
    m = (cfg.site == "L2.norm2") & (cfg.group == "all") & np.isclose(cfg.alpha, 1.0)
    if not ((cfg.method == "dm") & m).any():
        return None
    return {k: int(cfg[m & (cfg.method == k)].cid.iloc[0]) for k in ("dm", "rand")}


def stats_for(df: pd.DataFrame, cids: list, args) -> dict:
    """{cid: {group: metrics}} for the configs in cids on the boxes of df."""
    out = {c: {} for c in cids}
    img = df.image_name.to_numpy()
    for g in GROUPS:
        m = group_mask(df, g)
        sub_img = img[m]
        base = df.base_iou.to_numpy()[m]
        best = df.best_iou.to_numpy()[m]
        X = df[[f"iou_c{c}" for c in cids]].to_numpy()[m]
        H = df[[f"head_c{c}" for c in cids]].to_numpy()[m]
        h0 = df.base_head.to_numpy()[m]
        D = X - base[:, None]
        lo, hi = boot_ci(D, sub_img, args.n_boot, args.seed)
        gap = (best - base).mean()
        for j, c in enumerate(cids):
            d = D[:, j]
            out[c][g] = {"n": int(m.sum()), "base": float(base.mean()), "iou": float(X[:, j].mean()),
                         "d": float(d.mean()), "lo": float(lo[j]), "hi": float(hi[j]),
                         "pct": float(d.mean() / base.mean() * 100), "up": float((d > args.updown_eps).mean()),
                         "down": float((d < -args.updown_eps).mean()),
                         "gap": float(d.mean() / gap) if gap > 1e-9 else math.nan,
                         "hc": float((H[:, j] != h0).mean())}
    return out


def grid_table(R: dict) -> pd.DataFrame:
    rows = []
    cfg = R["cfg"]
    cols = [f"iou_c{c}" for c in cfg.cid]
    for mm in MODES:
        df = R["per"][mm]
        for ds in R["meta"]["datasets"]:
            sub = df[df.key == ds]
            for g in GROUPS:
                m = group_mask(sub, g)
                base = sub.base_iou.to_numpy()[m].mean()
                mean = sub[cols].to_numpy()[m].mean(0)
                for c, x in zip(cfg.cid, mean):
                    rows.append((mm, ds, g, int(c), float(x), float(x - base), float((x - base) / base * 100)))
    return pd.DataFrame(rows, columns=["mode", "dataset", "group", "cid", "iou", "d", "pct"])


def reference_table(R: dict, ref_path: str) -> dict:
    """Means of the reference csv's methods on the same boxes, and as % of the
    mode's baseline (our recomputed undef / head-select)."""
    if not ref_path or not Path(ref_path).is_file():
        return {}
    ref = pd.read_csv(ref_path, dtype={"image_name": str, "kind": str, "user": str, "attempt": str})
    ref[KEY[1:]] = ref[KEY[1:]].fillna("")
    cols = [c for c in REF_COLS if c in ref.columns]
    out = {}
    for mm in MODES:
        df = R["per"][mm].merge(ref[KEY + cols], on=KEY, how="left")
        out[mm] = {}
        for ds in R["meta"]["datasets"]:
            sub = df[df.key == ds]
            out[mm][ds] = {}
            for g in GROUPS:
                s = sub[group_mask(sub, g)]
                base = s.base_iou.mean()
                out[mm][ds][g] = {REF_COLS[c]: {"iou": float(s[c].mean()), "pct": float((s[c].mean() - base) / base * 100)}
                                  for c in cols if s[c].notna().all()}
    return out


def clean(o):
    """NaN / inf -> None, numpy -> python, for strict JSON."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def site_text(site: dict, layout: dict) -> dict:
    sid, k = site["id"], site["kind"]
    toks = ", ".join(layout["token_names"])
    eh, ew = layout["emb_hw"]
    if sid == "prompt_tok":
        title = "prompt tokens"
        text = (f"Выход prompt encoder: {layout['n_sparse']} токена ({', '.join(layout['token_names'][-layout['n_sparse']:])}). "
                "Вектор прибавляется ко всем; через повторное прибавление prompt-токенов он попадает в каждую attention декодера.")
    elif sid.startswith("L") and ".norm" in sid:
        li, n = sid[1], int(sid[-1])
        title = {1: f"слой {li}: после self-attention", 2: f"слой {li}: после token → image attention",
                 3: f"слой {li}: после MLP", 4: f"слой {li}: image embedding после image → token attention"}[n]
        text = {1: f"Токены декодера ({toks}) после self-attention и LayerNorm norm1"
                   + (": в слое 1 без residual." if li == "1" else "."),
                2: "Токены после того, как прочитали картинку (token → image attention), LayerNorm norm2. "
                   "У SAM-1 это узел, выбранный на critical shifts.",
                3: "Токены после поточечного MLP, LayerNorm norm3.",
                4: f"Image embedding ({eh} × {ew} клеток × 256) после image → token attention, LayerNorm norm4. "
                   "Один 256-мерный вектор прибавляется к клеткам, центр которых внутри бокса."}[n]
    elif sid == "final_norm":
        title = "после финальной token → image attention"
        text = ("Последнее состояние токенов перед выходными головами. Prompt-строки дальше не читаются, "
                "поэтому группы — все, mask token 0, mask tokens 1–3.")
    else:
        title = "выходы гиперсетей"
        text = (f"{layout['n_mask_tokens']} × 32 веса динамических классификаторов (по одному на mask token): "
                "логит пикселя = ⟨веса, upscaled embedding(x, y)⟩. У каждой гиперсети свой вектор.")
    return {"id": sid, "kind": k, "paths": site["paths"], "groups": site["groups"], "title": title, "text": text}


def build(args) -> int:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    R = load_run(run_dir)
    meta, cfg = R["meta"], R["cfg"]
    dsets = meta["datasets"]
    sel = select(R)
    fixed = fixed_cids(cfg)

    stats, sel_rows, site_rows = {}, [], []
    for mm in MODES:
        stats[mm] = {}
        cids = sorted({v for s in sel[mm]["sites"].values() for v in (s["dm"], s["rand"], s["oracle"])}
                      | (set(fixed.values()) if fixed else set()))
        for ds in dsets:
            st = stats_for(R["per"][mm][R["per"][mm].key == ds], cids, args)
            stats[mm][ds] = {"sites": {}, "fixed": None}
            for sid, s in sel[mm]["sites"].items():
                stats[mm][ds]["sites"][sid] = {m: st[s[m]] for m in ("dm", "rand", "oracle")}
                for m in ("dm", "rand", "oracle"):
                    for g, v in st[s[m]].items():
                        site_rows.append({"mode": mm, "dataset": ds, "site": sid, "method": m, "group": g,
                                          "cfg_group": s["group"] if m != "oracle" else WHOLE[next(x["kind"] for x in meta["sites"] if x["id"] == sid)],
                                          "alpha": s["alpha"] if m != "oracle" else math.nan, **v})
            if fixed:
                stats[mm][ds]["fixed"] = {m: st[fixed[m]] for m in ("dm", "rand")}
        for sid, s in sel[mm]["sites"].items():
            sel_rows.append({"mode": mm, "site": sid, **s, "best": sid == sel[mm]["best"]})
    grid = grid_table(R)
    grid.to_csv(out_dir / "grid.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(out_dir / "selected.csv", index=False)
    pd.DataFrame(site_rows).to_csv(out_dir / "table_sites.csv", index=False)

    expect = None
    # only a full run has the 25-image A vector of steer_user_study
    if meta["model_name"] == "SAM" and fixed and not meta["args"].get("limit_images"):
        expect = {}
        for ds, want in EXPECT_SAM1.items():
            if ds in dsets:
                got = stats[0][ds]["fixed"]["dm"]["all"]["d"]
                expect[ds] = {"want": want, "got": got, "ok": abs(got - want) <= 5e-4}
    refs = reference_table(R, args.reference_csv or meta["args"].get("reference_csv", ""))
    # head-select of this very run (the mm1 baseline; both csvs list the boxes in one order)
    p0, p1 = R["per"][0], R["per"][1]
    hs = p1.base_iou.to_numpy()
    for ds in dsets:
        for g in GROUPS:
            m = (p0.key.to_numpy() == ds) & group_mask(p0, g)
            base = p0.base_iou.to_numpy()[m].mean()
            refs.setdefault(0, {}).setdefault(ds, {}).setdefault(g, {})["head-select (этот прогон)"] = {
                "iou": float(hs[m].mean()), "pct": float((hs[m].mean() - base) / base * 100)}

    G = {mm: {ds: {} for ds in dsets} for mm in MODES}
    for r in grid.itertuples(index=False):
        G[r.mode][r.dataset].setdefault(r.cid, [None, None, None])[GROUPS.index(r.group)] = round(r.pct, 4)
    gridp = {mm: {ds: [G[mm][ds][c] for c in cfg.cid] for ds in dsets} for mm in MODES}
    probe = {}
    for r in R["probe"].itertuples(index=False):
        probe.setdefault(r.site, {})[r.dataset] = [r.auc, r.auc_within]
    vst = {}
    for r in R["vstats"].itertuples(index=False):
        vst.setdefault(r.site, {})[r.group] = {"dose": r.dose_alpha1, "fold_cos": r.fold_cos_mean,
                                               "v": r.v_norm, "h": r.h_norm}
    base_head = {mm: {ds: float((R["per"][mm][R["per"][mm].key == ds].base_head != 0).mean()) for ds in dsets}
                 for mm in MODES}
    layout = meta["layout"]
    payload = {
        "meta": {"model": meta["model_name"], "checkpoint": meta["checkpoint"], "created": meta["created"],
                 "report_created": time.strftime("%Y-%m-%d %H:%M"), "device": meta.get("device"),
                 "images": meta["images"], "boxes": meta["boxes"], "folds": meta["folds"], "alphas": meta["alphas"],
                 "versions": meta.get("versions", {}), "seconds": meta.get("seconds"), "updown": args.updown_eps},
        "layout": layout, "datasets": dsets, "spec": R["graph"],
        "sites": [site_text(s, layout) for s in meta["sites"]],
        "cfg": {"site": cfg.site.tolist(), "group": cfg.group.tolist(), "method": cfg.method.tolist(),
                "alpha": cfg.alpha.tolist(), "dose": cfg.dose.tolist()},
        "grid": gridp, "sel": sel, "stats": stats, "probe": probe, "vstats": vst, "refs": refs,
        "base_head": base_head, "checks": R["checks"], "expect": expect,
    }
    data = json.dumps(clean(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    out = Path(args.out) if args.out else out_dir / "steer_scan_report.html"
    out.write_text(HTML.replace("__PAYLOAD__", data), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_md(meta, sel, stats, dsets, expect), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {out_dir / 'summary.md'}")
    for mm in MODES:
        b = sel[mm]["best"]
        s = sel[mm]["sites"][b]
        print(f"mm{mm}: best site {b} ({s['group']}, alpha {s['alpha']:g}), A gain {s['score_A']:+.4f}; "
              + ", ".join(f"{ds} b50 {stats[mm][ds]['sites'][b]['dm']['bottom50']['pct']:+.2f}%" for ds in dsets))
    if expect:
        print("SAM-1 vs steer_user_study (L2.norm2, all, alpha 1, delta all): "
              + ", ".join(f"{k} {v['got']:+.4f} (want {v['want']:+.4f}) {'ok' if v['ok'] else 'MISMATCH'}"
                          for k, v in expect.items()))
    return 0


def summary_md(meta, sel, stats, dsets, expect) -> str:
    L = [f"# Steering scan: {meta['model_name']}", "",
         f"Прогон {meta['created']}, чекпоинт `{meta['checkpoint']}`; токены декодера: "
         f"{', '.join(meta['layout']['token_names'])}; fallback по стабильности: "
         f"{meta['layout']['dynamic_multimask_via_stability']}.", "",
         "Группа токенов и α выбраны на COCO A вне фолда (максимум ΔIoU по всем боксам); остальное заморожено. "
         "% — прирост к базе режима (mm0: один выход, mm1: head-select).", ""]
    for mm in MODES:
        L += [f"## multimask = {bool(mm)}  ·  лучший узел: {sel[mm]['best']}", "",
              "| узел | группа, α | " + " | ".join(f"{d} b50 / t50 / all" for d in dsets) + " |",
              "|---|---|" + "---|" * len(dsets)]
        for sid, s in sel[mm]["sites"].items():
            cells = []
            for d in dsets:
                v = stats[mm][d]["sites"][sid]["dm"]
                cells.append(" / ".join(f"{v[g]['pct']:+.2f}" for g in ("bottom50", "top50", "all")))
            name = f"**{sid}**" if sid == sel[mm]["best"] else sid
            L.append(f"| {name} | {s['group']}, {s['alpha']:g} | " + " | ".join(cells) + " |")
        L.append("")
    if expect:
        L += ["SAM-1 против steer_user_study (L2.norm2, все токены, α = 1, Δ all): "
              + ", ".join(f"{k} {v['got']:+.4f} (было {v['want']:+.4f})" for k, v in expect.items()), ""]
    return "\n".join(L)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="out_dir of steer_models_scan.py")
    p.add_argument("--out_dir", default="", help="default: --run_dir")
    p.add_argument("--out", default="", help="default: <out_dir>/steer_scan_report.html")
    p.add_argument("--reference_csv", default="", help="per-case csv with head-select / best-of-N / grad; "
                                                       "default: the run's --reference_csv")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updown_eps", type=float, default=0.01)
    return p.parse_args(argv)


def main(argv=None) -> int:
    return build(parse_args(argv))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steering scan</title>
<style>
:root{
  --scheme:light;
  --bg:#f9f9f7; --surface:#fcfcfb; --surface-2:#f0efec; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --sel:#0b0b0b; --link:#1c5cab;
  --ok:#006300; --warn:#b5520f; --fail:#d03b3b;
  --m-dm:#2a78d6; --m-rand:#1baf7a; --m-oracle:#52514e; --m-alt:#eb6834;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --scheme:dark;
    --bg:#0d0d0d; --surface:#1a1a19; --surface-2:#262624; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#4a4a46; --border:rgba(255,255,255,0.10); --sel:#ffffff; --link:#86b6ef;
    --ok:#0ca30c; --warn:#ec835a; --fail:#e66767;
    --m-dm:#3987e5; --m-rand:#199e70; --m-oracle:#c3c2b7; --m-alt:#d95926;
  }
}
:root[data-theme="dark"]{
  --scheme:dark;
  --bg:#0d0d0d; --surface:#1a1a19; --surface-2:#262624; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#4a4a46; --border:rgba(255,255,255,0.10); --sel:#ffffff; --link:#86b6ef;
  --ok:#0ca30c; --warn:#ec835a; --fail:#e66767;
  --m-dm:#3987e5; --m-rand:#199e70; --m-oracle:#c3c2b7; --m-alt:#d95926;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1600px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:16px;margin:0 0 6px}
h3{font-size:14px;margin:0 0 6px}
.meta{color:var(--ink-2);margin:0 0 10px;font-size:13px}
.mono,code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.muted{color:var(--muted)} .ok{color:var(--ok)} .warn{color:var(--warn)} .fail{color:var(--fail)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}
.chip{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:6px 10px;font-size:13px;max-width:560px}
.chip b{font-weight:650}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;padding:8px 0;border-bottom:1px solid var(--grid);margin-bottom:12px}
.seg{display:inline-flex;flex-wrap:wrap;border:1px solid var(--axis);border-radius:8px;overflow:hidden}
.seg button{font:inherit;font-size:13px;background:var(--surface);color:var(--ink);border:0;border-left:1px solid var(--axis);padding:5px 10px;cursor:pointer;display:inline-flex;align-items:center;gap:6px}
.seg button:first-child{border-left:0}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
.seg button:focus-visible,.linkbtn:focus-visible{outline:2px solid var(--link);outline-offset:-2px}
.lbl{font-size:12px;color:var(--ink-2);margin-right:6px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
.main{display:grid;grid-template-columns:auto minmax(0,1fr);gap:16px;align-items:start}
@media (max-width:1180px){.main{grid-template-columns:minmax(0,1fr)}}
.gwrap{position:sticky;top:58px}
@media (max-width:1180px){.gwrap{position:static}}
.legend{font-size:12px;color:var(--ink-2);margin:0 0 6px;display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;max-width:700px}
.legend .ramp{width:160px;height:9px;border-radius:3px;display:inline-block;vertical-align:middle;margin:0 6px}
.graph{overflow:auto;max-height:calc(100vh - 110px);width:fit-content;max-width:100%;background:var(--surface);border:1px solid var(--border);border-radius:10px}
@media (max-width:1180px){.graph{max-height:70vh}}
.graph svg{display:block}
.node.site{cursor:pointer}
.node:focus{outline:none}
.node:focus-visible rect.frame{stroke:var(--link);stroke-width:2.5}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin:0 0 14px}
.tag{display:inline-block;font-size:11.5px;border:1px solid var(--axis);border-radius:999px;padding:1px 8px;color:var(--ink-2);margin-right:6px}
.lead{margin:6px 0 10px;max-width:900px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink-2);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.cur td{background:var(--surface-2)}
tr.cur td:first-child{box-shadow:inset 3px 0 0 var(--ink)}
tr.pick{cursor:pointer}
tr.pick:hover td{background:var(--surface-2)}
.tbl{overflow-x:auto}
.sub{color:var(--muted);font-size:11px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(460px,100%),1fr));gap:18px}
.chart{min-width:0}
.chart svg{display:block;max-width:100%;overflow:visible}
.ckey{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink-2);margin:2px 0 4px}
.ckey span{display:inline-flex;align-items:center;gap:5px}
.ckey i{display:inline-block;width:16px;height:0;border-top:2px solid}
.note{color:var(--ink-2);font-size:12px;margin:4px 0 0}
.up{color:var(--ok)} .down{color:var(--fail)}
details{margin:14px 0;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px 12px}
summary{cursor:pointer;font-weight:600}
.gloss{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(330px,100%),1fr));gap:10px;margin:10px 0}
.term{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px 12px}
.term h3{margin:0 0 4px}
.term p{margin:4px 0}
.formula{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;background:var(--surface-2);border-radius:6px;padding:5px 8px;margin:5px 0;white-space:pre-wrap}
.tip{position:fixed;z-index:20;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:7px 10px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.18);max-width:380px}
.tip .v{font-variant-numeric:tabular-nums}
.tip .l{color:var(--ink-2)}
.linkbtn{background:none;border:0;padding:0;color:var(--link);font:inherit;font-size:13px;cursor:pointer;text-decoration:underline;text-underline-offset:2px}
.toks{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 0}
.tok{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11.5px;border:1px solid var(--axis);border-radius:5px;padding:1px 6px;background:var(--bg)}
.tok.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
</style>
</head>
<body>
<div class="wrap">
  <h1 id="h1">Steering</h1>
  <p class="meta" id="meta"></p>
  <div class="chips" id="chips"></div>
  <div class="bar" role="toolbar" aria-label="Режим отчёта">
    <div><span class="lbl">Выход</span><span class="seg" id="seg-mm"></span></div>
    <div><span class="lbl">Данные</span><span class="seg" id="seg-ds"></span></div>
    <div><span class="lbl">Метод</span><span class="seg" id="seg-m"></span></div>
    <div><span class="lbl">Боксы</span><span class="seg" id="seg-g"></span></div>
    <div><span class="lbl">Цвет узлов</span><span class="seg" id="seg-c"></span></div>
    <button type="button" class="linkbtn" id="btn-terms">термины ↓</button>
  </div>
  <div class="main">
    <div class="gwrap">
      <div class="legend" id="legend"></div>
      <div class="graph" id="graph"></div>
    </div>
    <div>
      <section class="card" id="site"></section>
      <section class="card">
        <div class="charts">
          <div class="chart" id="ch-layers"></div>
          <div class="chart" id="ch-dose"></div>
        </div>
      </section>
    </div>
  </div>
  <section class="card" id="overview"></section>
  <section class="card" id="methods"></section>
  <details id="terms">
    <summary>Термины</summary>
    <div class="gloss">
      <div class="term"><h3>Боксы и группы</h3><p>User study FOR_TEST, мобильная разметка: на каждой картинке 25 человек обвели объект; бокс — tight bbox их маски. Внутри картинки боксы ранжируются по IoU одного выхода модели (multimask=False) без вмешательства: 12 худших — <b>bottom50</b>, 13 лучших — <b>top50</b>. Группы одни и те же в обоих режимах выхода и нужны только для отчёта.</p></div>
      <div class="term"><h3>Вектор DM</h3><div class="formula">d_img = среднее(top50) − среднее(bottom50)   (внутри картинки)
v = среднее d_img по картинкам COCO A
h' = h + α·v</div><p>Разница берётся внутри картинки, чтобы в вектор не попало содержимое картинок. У каждого токена свой вектор; в image stream — один 256-мерный вектор на клетки внутри бокса.</p></div>
      <div class="term"><h3>Вне фолда и перенос</h3><p><b>COCO A</b>: 25 картинок, 5 фолдов; картинка получает вектор остальных фолдов. По этим числам выбираются группа токенов и α (максимум ΔIoU по всем боксам A). <b>COCO B, Berkeley, GrabCut</b>: вектор по всему A, выбор заморожен — это честные тестовые числа.</p></div>
      <div class="term"><h3>Случайное направление</h3><p>Вектор той же нормы, что DM (на каждом токене), в фиксированном случайном направлении, с теми же группой и α. Если оно помогает так же, дело в толчке, а не в направлении.</p></div>
      <div class="term"><h3>Oracle (patching)</h3><p>Каждый бокс получает тензор узла целиком от <b>лучшего бокса той же картинки</b> (максимум IoU среди 25). Это подсмотренный ответ: потолок того, что можно исправить в этом узле. В prompt tokens oracle = подменить весь бокс.</p></div>
      <div class="term"><h3>Режимы выхода</h3><p><b>multimask=False</b> — один выход модели. У SAM-1 это mask token 0; у SAM-2.1 и SAM-3 декодер подменяет его лучшей из масок 1–3, если маска token 0 нестабильна (fallback по стабильности) — так модель поставляется. <b>multimask=True</b> — три маски и голова с максимальным предсказанным IoU (head-select). Поток токенов одинаков в обоих режимах; режим лишь выбирает, какой выход станет маской.</p></div>
      <div class="term"><h3>Метрики</h3><div class="formula">Δ = IoU(с вектором) − IoU(база режима)
% = Δ / IoU(база) · 100
разрыв закрыт = Δ / (IoU лучшего бокса − IoU базы)</div><p>CI — 95% парный бутстрап по картинкам (боксы картинки вместе). ↑ / ↓ — доля боксов, где IoU вырос / упал больше чем на порог. «Смена головы» — у какой доли боксов выход стал браться с другого mask token.</p></div>
      <div class="term"><h3>Доза и AUC пробы</h3><p><b>Доза</b> = ‖α·v‖ / средняя ‖h‖ на боксах A в этих строках (для image stream — по усреднённому по боксу состоянию). <b>AUC пробы</b>: логистическая регрессия на всём тензоре узла отличает top50 от bottom50; на A — вне фолда, на тестах — проба по всему A. «Внутри картинки» — средний AUC по картинкам: различает ли проба людей на одном и том же объекте.</p></div>
    </div>
  </details>
  <details id="checks"><summary>Проверки</summary><div id="checks-body"></div></details>
</div>
<div class="tip" id="tip" hidden></div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
"use strict";
(() => {
const D = JSON.parse(document.getElementById("payload").textContent);
const S = D.spec, LY = D.layout;
const NODE = Object.fromEntries(S.nodes.map(n => [n.id, n]));
const SITE = Object.fromEntries(D.sites.map(s => [s.id, s]));
const SVGNS = "http://www.w3.org/2000/svg";
const LANE_W = 168, NODE_W = 152, NODE_H = 40, ROW_H = 50, GROUP_GAP = 30, PAD = 10, HEAD = 22;
const M = {
  dm:     {name: "DM", long: "разность средних top50 − bottom50", cv: "--m-dm"},
  rand:   {name: "Случайное", long: "контроль: случайное направление нормы DM", cv: "--m-rand"},
  oracle: {name: "Oracle", long: "patching: тензор от лучшего бокса картинки", cv: "--m-oracle"},
};
const GL = {all: "все", bottom50: "bottom50", top50: "top50"};
const TGL = {all: "все токены", mask0: "mask token 0", mask123: "mask tokens 1–3", prompt: "prompt-токены", box: "клетки в боксе"};
const DSN = ds => ds === "COCO_A" ? "COCO A · вне фолда" : ds === "COCO_B" ? "COCO B" : ds;
const isFit = ds => ds === "COCO_A";
const tests = D.datasets.filter(d => !isFit(d));
const st = {mm: "0", ds: tests[0] || D.datasets[0], m: "dm", g: "bottom50", c: "d", site: null};
st.site = D.sel[st.mm].best;
const $ = id => document.getElementById(id);

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else if (k === "style") e.style.cssText = v; else e.setAttribute(k, v); }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}
function sv(tag, attrs) { const e = document.createElementNS(SVGNS, tag); if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; }
function svText(x, y, text, attrs, fill) { const t = sv("text", Object.assign({x, y}, attrs || {})); t.textContent = text; t.style.fill = fill || "var(--ink-2)"; return t; }
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const mcol = m => css(M[m].cv);
const ok = x => x != null && isFinite(x);
const minus = s => s.replace("-", "−");
const fp = (x, d = 2) => !ok(x) ? "—" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d) + "%";
const fd = (x, d = 4) => !ok(x) ? "—" : Math.abs(x) < 0.5 * Math.pow(10, -d) ? (0).toFixed(d) : (x > 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const f3 = x => ok(x) ? x.toFixed(4) : "—";
const share = x => ok(x) ? Math.round(x * 100) + "%" : "—";
const alphaT = a => ok(a) ? String(a) : "—";
const chartW = box => Math.max(320, Math.floor(box.clientWidth || 520));

// ---------- data ----------
const sel = () => D.sel[st.mm];
const chosen = sid => sel().sites[sid];
function stat(ds, sid, m, g) { const x = (((D.stats[st.mm] || {})[ds] || {}).sites || {})[sid]; return x && x[m] ? x[m][g] : null; }
function pctOf(ds, sid, m, g) { const v = stat(ds, sid, m, g); return v ? v.pct : null; }
function grid(ds, cid, g) { const r = ((D.grid[st.mm] || {})[ds] || [])[cid]; return r ? r[["all", "bottom50", "top50"].indexOf(g)] : null; }
function aucs(ds, sid) { const p = (D.probe[sid] || {})[ds]; return p || [null, null]; }

// ---------- colour ----------
const SEQ = {light: ["#e9f1fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"],
             dark:  ["#1f2a3a", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"]};
const NEG = {light: ["#fdf0e9", "#f8c8ad", "#f29a6c", "#eb6834", "#b5520f", "#7a3608"],
             dark:  ["#3a2418", "#7a3a17", "#b5520f", "#d95926", "#ec835a", "#f6c3a8"]};
const scheme = () => css("--scheme") === "dark" ? "dark" : "light";
function hex2rgb(h) { h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
function rgb2hex(c) { return "#" + c.map(v => Math.round(v).toString(16).padStart(2, "0")).join(""); }
function rampOf(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[i + 1]);
  return rgb2hex(a.map((v, k) => v + (b[k] - v) * f));
}
const ramp = t => rampOf(SEQ[scheme()], t);
const div = t => t >= 0 ? rampOf(SEQ[scheme()], t) : rampOf(NEG[scheme()], -t);
function textOn(hex) {
  const [r, g, b] = hex2rgb(hex).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 ? "#0b0b0b" : "#ffffff";
}
function scaleD() {
  const v = D.sites.map(s => Math.abs(pctOf(st.ds, s.id, st.m, st.g) || 0));
  return Math.max(1, ...v);
}
function nodeValue(sid) {
  if (st.c === "auc") { const a = aucs(st.ds, sid)[0]; return ok(a) ? {fill: ramp((a - 0.5) / 0.5), label: a.toFixed(2)} : null; }
  const v = pctOf(st.ds, sid, st.m, st.g); return ok(v) ? {fill: div(v / scaleD()), label: fp(v, 1)} : null;
}

// ---------- tooltip ----------
const tip = $("tip");
function showTip(ev, rows) { tip.replaceChildren(...rows.map(([c, t]) => el("div", c ? {class: c} : null, t))); tip.hidden = false; moveTip(ev); }
function moveTip(ev) {
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = innerHeight - r.height - 8;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };
function nodeTip(id) {
  const n = NODE[id], s = SITE[id];
  const rows = [["", (s ? id + " · " : "") + n.label], ["l", n.code]];
  if (n.note) rows.push(["l", n.note]);
  if (!s) { rows.push(["l", n.kind === "const" ? "не зависит от бокса" : "сюда вектор не добавлялся"]); return rows; }
  const c = chosen(id);
  rows.push(["l", "выбор на COCO A: " + TGL[c.group] + ", α = " + alphaT(c.alpha)]);
  for (const m of ["dm", "rand", "oracle"]) rows.push(["v", M[m].name + " · " + GL[st.g] + ": " + fp(pctOf(st.ds, id, m, st.g))]);
  const a = aucs(st.ds, id); rows.push(["v", "AUC пробы: " + (ok(a[0]) ? a[0].toFixed(2) : "—") + " · внутри картинки " + (ok(a[1]) ? a[1].toFixed(2) : "—")]);
  return rows;
}

// ---------- graph ----------
const SUCC = {};
for (const n of S.nodes) SUCC[n.id] = [];
for (const e of S.edges) SUCC[e.src].push(e.dst);
function reach(id) { const seen = new Set([id]), stack = [id]; while (stack.length) for (const y of SUCC[stack.pop()]) if (!seen.has(y)) { seen.add(y); stack.push(y); } return seen; }
function layout() {
  const starts = new Set(S.groups.map(g => g.row0));
  const maxRow = Math.max(...S.nodes.map(n => n.row));
  const rowY = []; let acc = HEAD;
  for (let r = 0; r <= maxRow; r++) { if (starts.has(r)) acc += GROUP_GAP; rowY.push(PAD + r * ROW_H + acc); }
  const pos = {};
  for (const n of S.nodes) pos[n.id] = {x: PAD + n.lane * LANE_W + (LANE_W - NODE_W) / 2, y: rowY[n.row], w: NODE_W, h: NODE_H};
  return {rowY, W: 2 * PAD + S.lanes.length * LANE_W, H: rowY[maxRow] + NODE_H + PAD + 8, pos};
}
function ports(edges, pos) {
  const inc = {}, out = {}, pin = {}, pout = {};
  for (const e of edges) { if (pos[e.src].y === pos[e.dst].y) continue; (inc[e.dst] = inc[e.dst] || []).push(e); (out[e.src] = out[e.src] || []).push(e); }
  for (const es of Object.values(inc)) { es.sort((a, b) => pos[a.src].x - pos[b.src].x || pos[a.src].y - pos[b.src].y); es.forEach((e, i) => pin[e.id] = (i - (es.length - 1) / 2) * 14); }
  for (const es of Object.values(out)) { es.sort((a, b) => pos[a.dst].x - pos[b.dst].x || pos[a.dst].y - pos[b.dst].y); es.forEach((e, i) => pout[e.id] = (i - (es.length - 1) / 2) * 14); }
  return {pin, pout};
}
function edgePath(e, pos, pp) {
  const a = pos[e.src], b = pos[e.dst];
  if (a.y === b.y) { const right = a.x < b.x, sx = right ? a.x + a.w : a.x, ex = right ? b.x - 3 : b.x + b.w + 3, y = a.y + a.h / 2; return `M${sx},${y} L${ex},${y}`; }
  const sx = a.x + a.w / 2 + (pp.pout[e.id] || 0), sy = a.y + a.h, ex = b.x + b.w / 2 + (pp.pin[e.id] || 0), ey = b.y - 3;
  const k = Math.min(30, (ey - sy) / 2);
  return `M${sx},${sy} C${sx},${sy + k} ${ex},${ey - k} ${ex},${ey}`;
}
function drawGraph(scroll) {
  const L = layout();
  const svg = sv("svg", {width: L.W, height: L.H, viewBox: `0 0 ${L.W} ${L.H}`, role: "img", "aria-label": "Граф декодера " + D.meta.model + ": узлы, куда добавляется вектор, окрашены"});
  const defs = sv("defs");
  for (const [id, col] of [["arr", "var(--axis)"], ["arr-on", "var(--ink-2)"]]) {
    const mk = sv("marker", {id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse"});
    const p = sv("path", {d: "M0,1 L9,5 L0,9 z"}); p.style.fill = col; mk.appendChild(p); defs.appendChild(mk);
  }
  const pat = sv("pattern", {id: "hatch", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"});
  const pr = sv("rect", {width: 7, height: 7}); pr.style.fill = "var(--surface-2)"; pat.appendChild(pr);
  const pl = sv("line", {x1: 0, y1: 0, x2: 0, y2: 7, "stroke-width": 2}); pl.style.stroke = "var(--grid)"; pat.appendChild(pl);
  defs.appendChild(pat); svg.appendChild(defs);
  S.lanes.forEach((l, i) => svg.appendChild(svText(PAD + i * LANE_W + LANE_W / 2, PAD + 10, l.label, {"text-anchor": "middle", "font-size": 11}, "var(--muted)")));
  for (const g of S.groups) {
    const x = PAD + g.lane0 * LANE_W + 3, y = L.rowY[g.row0] - GROUP_GAP + 5;
    const r = sv("rect", {x, y, width: (g.lane1 - g.lane0 + 1) * LANE_W - 6, height: L.rowY[g.row1] + NODE_H + 8 - y, rx: 9});
    r.style.fill = "none"; r.style.stroke = "var(--grid)"; r.style.strokeWidth = 1.5; svg.appendChild(r);
    const t = svText(x + 9, y + 15, g.label, {"font-size": 11.5, "font-weight": 600});
    t.style.stroke = "var(--surface)"; t.style.strokeWidth = 4; t.style.paintOrder = "stroke"; svg.appendChild(t);
  }
  const down = reach(st.site);
  const edges = S.edges.filter(e => e.role !== "pe");
  const pp = ports(edges, L.pos);
  const eg = sv("g");
  for (const e of edges) {
    const on = down.has(e.src);
    const p = sv("path", {d: edgePath(e, L.pos, pp), fill: "none", "marker-end": on ? "url(#arr-on)" : "url(#arr)"});
    p.style.stroke = on ? "var(--ink-2)" : "var(--axis)"; p.style.strokeWidth = on ? 1.6 : 1.1;
    if (!on) p.style.opacity = 0.55;
    if (NODE[e.src].kind === "const") p.setAttribute("stroke-dasharray", "4 3");
    eg.appendChild(p);
  }
  svg.appendChild(eg);
  let selY = 0;
  for (const n of S.nodes) {
    const p = L.pos[n.id], site = SITE[n.id], isSel = n.id === st.site;
    const g = sv("g", {class: "node" + (site ? " site" : ""), transform: `translate(${p.x},${p.y})`});
    if (site) { g.setAttribute("tabindex", 0); g.setAttribute("role", "button"); g.setAttribute("aria-pressed", isSel); g.setAttribute("aria-label", n.id + ": " + n.label); }
    if (!site && !down.has(n.id)) g.style.opacity = 0.5;
    const r = sv("rect", {class: "frame", width: p.w, height: p.h, rx: 7});
    let ink = "var(--ink)", nv = null;
    if (site) {
      nv = nodeValue(n.id);
      const fill = nv ? nv.fill : "var(--surface-2)";
      r.style.fill = fill; if (nv) ink = textOn(fill);
      r.style.stroke = isSel ? "var(--sel)" : "var(--ink-2)"; r.style.strokeWidth = isSel ? 3 : 1.2;
    } else { r.style.fill = n.kind === "const" ? "url(#hatch)" : "var(--surface)"; r.style.stroke = "var(--grid)"; r.style.strokeWidth = 1; }
    g.appendChild(r);
    g.appendChild(svText(8, 16, n.label, {"font-size": 11.5, "font-weight": site ? 650 : 500}, site ? ink : "var(--ink-2)"));
    if (site) {
      g.appendChild(svText(8, 32, n.id, {"font-size": 10.5}, ink));
      g.appendChild(svText(p.w - 8, 32, nv ? nv.label : "—", {"font-size": 12, "font-weight": 650, "text-anchor": "end"}, ink));
      if (n.id === sel().best) {
        const star = svText(p.w - 8, 16, "★", {"font-size": 12, "text-anchor": "end"}, ink);
        g.appendChild(star);
      }
    }
    if (isSel) {
      selY = p.y;
      const b = sv("g", {transform: `translate(${-6},${p.h / 2})`});
      const c = sv("rect", {x: -36, y: -10, width: 36, height: 20, rx: 10}); c.style.fill = "var(--sel)"; b.appendChild(c);
      b.appendChild(svText(-18, 4, st.m === "oracle" ? "patch" : "+αv", {"font-size": 11, "font-weight": 700, "text-anchor": "middle"}, "var(--bg)"));
      g.appendChild(b);
    }
    g.addEventListener("pointerenter", ev => showTip(ev, nodeTip(n.id)));
    g.addEventListener("pointermove", moveTip);
    g.addEventListener("pointerleave", hideTip);
    if (site) {
      g.addEventListener("click", () => select(n.id));
      g.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(n.id); } });
    }
    svg.appendChild(g);
  }
  const box = $("graph");
  box.replaceChildren(svg);
  if (scroll) box.scrollTo({top: Math.max(0, selY - box.clientHeight / 3), behavior: "smooth"});
}
function drawLegend() {
  const lg = $("legend"); lg.replaceChildren();
  if (st.c === "auc") {
    const stops = Array.from({length: 11}, (_, i) => ramp(i / 10) + " " + i * 10 + "%").join(",");
    lg.append(el("span", null, "AUC пробы top50 vs bottom50: 0.5", el("span", {class: "ramp", style: "background:linear-gradient(90deg," + stops + ")"}), "1.0"));
  } else {
    const s = scaleD();
    const stops = Array.from({length: 21}, (_, i) => div(i / 10 - 1) + " " + i * 5 + "%").join(",");
    lg.append(el("span", null, M[st.m].name + ", " + GL[st.g] + ", % к базе: " + fp(-s, 1), el("span", {class: "ramp", style: "background:linear-gradient(90deg," + stops + ")"}), fp(s, 1)));
  }
  lg.append(el("span", null, "★ — лучший узел по COCO A; клик по цветному узлу — выбрать его; тёмные рёбра — куда вектор распространяется дальше"));
}

// ---------- charts ----------
function axisY(svg, x0, x1, y, ticks, fmt) {
  for (const t of ticks) {
    const yy = y(t);
    const l = sv("line", {x1: x0, x2: x1, y1: yy, y2: yy}); l.style.stroke = t === 0 ? "var(--axis)" : "var(--grid)"; l.style.strokeWidth = 1; svg.appendChild(l);
    svg.appendChild(svText(x0 - 6, yy + 4, fmt(t), {"font-size": 11, "text-anchor": "end"}, "var(--muted)"));
  }
}
function niceTicks(lo, hi, n) {
  const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => span / s <= n) || 10 * mag;
  const out = []; for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.abs(t) < 1e-12 ? 0 : t);
  return out;
}
function dot(svg, x, y, color, rows, onclick, big) {
  const c = sv("circle", {cx: x, cy: y, r: big ? 5 : 3.6}); c.style.fill = color; c.style.stroke = "var(--surface)"; c.style.strokeWidth = 1.5; svg.appendChild(c);
  const h = sv("circle", {cx: x, cy: y, r: 10}); h.style.fill = "transparent"; if (onclick) h.style.cursor = "pointer";
  h.addEventListener("pointerenter", ev => showTip(ev, rows)); h.addEventListener("pointermove", moveTip); h.addEventListener("pointerleave", hideTip);
  if (onclick) h.addEventListener("click", onclick);
  svg.appendChild(h);
}
function poly(svg, pts, color, dash, width) {
  const segs = []; let cur = [];
  for (const p of pts) { if (p && ok(p[1])) cur.push(p); else { if (cur.length) segs.push(cur); cur = []; } }
  if (cur.length) segs.push(cur);
  for (const s of segs) {
    const p = sv("path", {d: "M" + s.map(q => q[0].toFixed(1) + "," + q[1].toFixed(1)).join(" L"), fill: "none"});
    p.style.stroke = color; p.style.strokeWidth = width || 2; p.style.strokeLinejoin = "round"; if (dash) p.setAttribute("stroke-dasharray", dash);
    svg.appendChild(p);
  }
}
function key(items) { return el("div", {class: "ckey"}, ...items.map(([label, color, dash]) => el("span", null, el("i", {style: `border-top:2px ${dash ? "dashed" : "solid"} ${color}`}), label))); }
function drawLayers() {
  const box = $("ch-layers"); box.replaceChildren();
  const ids = D.sites.map(s => s.id), ds = st.ds, g = st.g;
  const lines = ["dm", "rand", "oracle"];
  box.append(el("h3", null, "Эффект по узлам · " + DSN(ds) + " · " + GL[g]));
  box.append(key(lines.map(m => [M[m].name, mcol(m), m === "oracle"]).concat([["AUC пробы", css("--ink-2"), false], ["AUC внутри картинки", css("--ink-2"), true]])));
  const W = chartW(box), ml = 46, mr = 10, mt = 8, h1 = 190, gap = 30, h2 = 66, mb = 62;
  const svg = sv("svg", {width: W, height: mt + h1 + gap + h2 + mb, viewBox: `0 0 ${W} ${mt + h1 + gap + h2 + mb}`, role: "img", "aria-label": "Прирост IoU по узлам и AUC пробы"});
  const x = i => ml + (i + 0.5) * (W - ml - mr) / ids.length;
  const vals = lines.flatMap(m => ids.map(s => pctOf(ds, s, m, g))).filter(ok);
  let lo = Math.min(0, ...vals), hi = Math.max(0.5, ...vals);
  if (hi - lo < 1) hi = lo + 1;
  const y = v => mt + h1 - (v - lo) / (hi - lo) * h1;
  const si = ids.indexOf(st.site), bw = (W - ml - mr) / ids.length;
  const band = sv("rect", {x: x(si) - bw / 2, y: mt, width: bw, height: h1 + gap + h2}); band.style.fill = "var(--surface-2)"; svg.appendChild(band);
  axisY(svg, ml, W - mr, y, niceTicks(lo, hi, 5), t => fp(t, Math.abs(hi - lo) < 4 ? 1 : 0));
  for (const m of lines) poly(svg, ids.map((s, i) => [x(i), ok(pctOf(ds, s, m, g)) ? y(pctOf(ds, s, m, g)) : null]), mcol(m), m === "oracle" ? "5 4" : null, m === st.m ? 2.6 : 1.8);
  for (const m of lines) ids.forEach((s, i) => {
    const v = stat(ds, s, m, g); if (!v || !ok(v.pct)) return;
    const c = chosen(s);
    dot(svg, x(i), y(v.pct), mcol(m), [["", s + " · " + M[m].name], ["v", fp(v.pct) + " · Δ " + fd(v.d) + (ok(v.lo) ? " [" + fd(v.lo) + ", " + fd(v.hi) + "]" : "")],
      ["l", m === "oracle" ? "тензор целиком от лучшего бокса" : TGL[c.group] + ", α = " + alphaT(c.alpha)]], () => select(s), m === st.m && s === st.site);
  });
  const y2 = v => mt + h1 + gap + h2 - (v - 0.5) / 0.5 * h2;
  axisY(svg, ml, W - mr, y2, [0.5, 0.75, 1], t => t.toFixed(2));
  svg.appendChild(svText(ml, mt + h1 + gap - 8, "AUC пробы top50 vs bottom50", {"font-size": 11}, "var(--ink-2)"));
  for (const [k, dash] of [[0, null], [1, "4 3"]]) {
    poly(svg, ids.map((s, i) => { const a = aucs(ds, s)[k]; return [x(i), ok(a) ? y2(Math.max(0.5, a)) : null]; }), "var(--ink-2)", dash, 1.6);
    ids.forEach((s, i) => { const a = aucs(ds, s)[k]; if (ok(a)) dot(svg, x(i), y2(Math.max(0.5, a)), "var(--ink-2)", [["", s], ["v", (k ? "AUC внутри картинки " : "AUC ") + a.toFixed(3)]], () => select(s)); });
  }
  ids.forEach((s, i) => {
    const t = svText(x(i) + 3, mt + h1 + gap + h2 + 14, s, {"font-size": 11, "text-anchor": "end", transform: `rotate(-40 ${x(i) + 3} ${mt + h1 + gap + h2 + 14})`}, s === st.site ? "var(--ink)" : "var(--muted)");
    t.style.cursor = "pointer"; if (s === st.site) t.setAttribute("font-weight", 700);
    t.addEventListener("click", () => select(s)); svg.appendChild(t);
  });
  box.append(svg);
  box.append(el("p", {class: "note"}, "Для DM и случайного направления — группа токенов и α, выбранные на COCO A вне фолда; oracle — подмена тензора целиком от лучшего бокса картинки."));
}
function drawDose() {
  const box = $("ch-dose"); box.replaceChildren();
  const sid = st.site, c = chosen(sid), ds = st.ds;
  const rows = D.cfg.site.map((s, i) => i).filter(i => D.cfg.site[i] === sid && D.cfg.group[i] === c.group && D.cfg.method[i] !== "oracle");
  box.append(el("h3", null, "Доза → эффект в " + sid + " · " + TGL[c.group] + " · " + DSN(ds)));
  const series = [["dm", "bottom50", null], ["dm", "top50", "5 4"], ["rand", "bottom50", null], ["rand", "top50", "5 4"]];
  box.append(key(series.map(([m, g, dash]) => [M[m].name + " · " + GL[g], mcol(m), !!dash])));
  const pts = {}; for (const [m, g] of series) pts[m + g] = rows.filter(i => D.cfg.method[i] === m).sort((a, b) => D.cfg.alpha[a] - D.cfg.alpha[b]).map(i => [D.cfg.alpha[i], grid(ds, i, g), D.cfg.dose[i]]);
  const all = Object.values(pts).flat();
  if (!all.length) { box.append(el("p", {class: "note"}, "нет данных")); return; }
  const W = chartW(box), H = 300, ml = 46, mr = 12, mt = 8, mb = 40;
  const svg = sv("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Прирост IoU от силы α"});
  const as = all.map(p => p[0]), lx0 = Math.log2(Math.min(...as)), lx1 = Math.max(lx0 + 1, Math.log2(Math.max(...as)));
  const ys = all.map(p => p[1]).filter(ok);
  let lo = Math.min(0, ...ys), hi = Math.max(0.5, ...ys); if (hi - lo < 1) hi = lo + 1;
  const x = a => ml + (Math.log2(a) - lx0) / (lx1 - lx0) * (W - ml - mr);
  const y = v => mt + (H - mt - mb) - (v - lo) / (hi - lo) * (H - mt - mb);
  axisY(svg, ml, W - mr, y, niceTicks(lo, hi, 6), t => fp(t, Math.abs(hi - lo) < 4 ? 1 : 0));
  for (const a of [...new Set(as)]) {
    const xx = x(a), l = sv("line", {x1: xx, x2: xx, y1: mt, y2: H - mb}); l.style.stroke = "var(--grid)"; l.style.strokeWidth = a === c.alpha ? 0 : 0.6; svg.appendChild(l);
    svg.appendChild(svText(xx, H - mb + 14, String(a), {"font-size": 10.5, "text-anchor": "middle"}, a === c.alpha ? "var(--ink)" : "var(--muted)"));
  }
  const cx = x(c.alpha), cl = sv("line", {x1: cx, x2: cx, y1: mt, y2: H - mb}); cl.style.stroke = "var(--ink-2)"; cl.setAttribute("stroke-dasharray", "3 3"); svg.appendChild(cl);
  svg.appendChild(svText(cx + 4, mt + 10, "выбор на A", {"font-size": 10.5}, "var(--ink-2)"));
  svg.appendChild(svText(ml + (W - ml - mr) / 2, H - 6, "α (лог. шкала)", {"font-size": 11, "text-anchor": "middle"}, "var(--ink-2)"));
  for (const [m, g, dash] of series) poly(svg, pts[m + g].map(p => [x(p[0]), ok(p[1]) ? y(p[1]) : null]), mcol(m), dash, m === "dm" ? 2.4 : 1.6);
  for (const [m, g] of series) for (const p of pts[m + g]) if (ok(p[1]))
    dot(svg, x(p[0]), y(p[1]), mcol(m), [["", M[m].name + " · " + GL[g] + " · α = " + p[0]], ["v", fp(p[1]) + " к базе"], ["l", "доза ‖αv‖/‖h‖ = " + (ok(p[2]) ? (p[2] * 100).toPrecision(2) + "%" : "—")]]);
  box.append(svg);
  box.append(el("p", {class: "note"}, "Сплошные — bottom50, пунктир — top50. Хороший узел: DM поднимает bottom50 и не опускает top50, случайное направление около нуля."));
}

// ---------- site panel ----------
function tokenChips(sid) {
  const s = SITE[sid], c = chosen(sid);
  if (s.kind !== "tok" && s.kind !== "sparse") return null;
  const rows = s.kind === "sparse" ? LY.rows.prompt : LY.rows[c.group];
  const all = s.kind === "sparse" ? LY.rows.prompt : LY.rows.all;
  return el("div", {class: "toks"}, ...all.map(i => el("span", {class: "tok" + (rows.includes(i) ? " on" : "")}, i + " " + LY.token_names[i])));
}
function drawSite() {
  const s = SITE[st.site], ds = st.ds, box = $("site"), c = chosen(s.id); box.replaceChildren();
  box.append(el("h2", null, s.id + " · " + s.title + (s.id === sel().best ? "  ★" : "")));
  box.append(el("div", null, el("span", {class: "tag"}, {sparse: "prompt encoder", tok: "поток токенов", img: "image stream", vec: "выходная голова"}[s.kind]),
    el("span", {class: "mono muted"}, s.paths.length > 1 ? s.paths[0] + " … " + s.paths[s.paths.length - 1] : s.paths[0])));
  box.append(el("p", {class: "lead"}, s.text));
  const vs = (D.vstats[s.id] || {})[c.group] || {};
  box.append(el("p", null, "Выбор на COCO A вне фолда: ", el("b", null, TGL[c.group] + ", α = " + alphaT(c.alpha)),
    " (доза " + (ok(c.dose) ? (c.dose * 100).toPrecision(2) + "% ‖h‖" : "—") + "; ΔIoU всех боксов A " + fd(c.score_A) + "). Устойчивость вектора: косинус между векторами фолдов " + (ok(vs.fold_cos) ? vs.fold_cos.toFixed(2) : "—") + "."));
  const tc = tokenChips(s.id); if (tc) box.append(tc);
  const v = stat(ds, s.id, "dm", st.g), r = stat(ds, s.id, "rand", st.g), o = stat(ds, s.id, "oracle", st.g);
  if (v) box.append(el("p", null, DSN(ds) + (isFit(ds) ? "" : " (вектор и выбор заморожены по COCO A)") + ", " + GL[st.g] + ": DM " + fp(v.pct) + " [Δ " + fd(v.lo) + " … " + fd(v.hi) + "]" +
    (r ? "; случайное направление " + fp(r.pct) : "") + (o ? "; потолок oracle " + fp(o.pct) : "") + (ok(v.gap) && st.g !== "top50" ? "; закрыто " + share(v.gap) + " разрыва до лучшего бокса" : "") + "."));
  const tb = el("table");
  tb.append(el("thead", null, el("tr", null, ...["Боксы", "n", "База", "DM", "Δ [95% CI]", "%", "Случайное", "Oracle", "↑ / ↓", "Разрыв закрыт", "Смена головы"].map((h, i) => el("th", {class: i ? "num" : ""}, h)))));
  const body = el("tbody");
  for (const g of ["all", "bottom50", "top50"]) {
    const a = stat(ds, s.id, "dm", g), b = stat(ds, s.id, "rand", g), q = stat(ds, s.id, "oracle", g);
    if (!a) continue;
    const tr = el("tr", {class: "pick" + (g === st.g ? " cur" : "")},
      el("td", null, GL[g]), el("td", {class: "num"}, String(a.n)), el("td", {class: "num"}, f3(a.base)), el("td", {class: "num"}, f3(a.iou)),
      el("td", {class: "num"}, fd(a.d), el("div", {class: "sub"}, fd(a.lo) + " … " + fd(a.hi))), el("td", {class: "num"}, fp(a.pct)),
      el("td", {class: "num"}, b ? fp(b.pct) : "—"), el("td", {class: "num"}, q ? fp(q.pct) : "—"),
      el("td", {class: "num"}, el("span", {class: "up"}, share(a.up)), " / ", el("span", {class: "down"}, share(a.down))),
      el("td", {class: "num"}, share(a.gap)), el("td", {class: "num"}, share(a.hc)));
    tr.tabIndex = 0; tr.addEventListener("click", () => setG(g)); tr.addEventListener("keydown", ev => { if (ev.key === "Enter") setG(g); });
    body.append(tr);
  }
  tb.append(body);
  box.append(el("div", {class: "tbl"}, tb));
  const a = aucs(ds, s.id);
  box.append(el("p", {class: "note"}, "База режима: " + (st.mm === "0" ? "один выход (multimask=False)" : "head-select (multimask=True)") + ". AUC пробы top50 vs bottom50: " + (ok(a[0]) ? a[0].toFixed(2) : "—") + ", внутри картинки " + (ok(a[1]) ? a[1].toFixed(2) : "—") + ". ↑ / ↓ — IoU изменился больше чем на " + D.meta.updown + "."));
}

// ---------- overview, methods ----------
function drawOverview() {
  const box = $("overview"); box.replaceChildren();
  box.append(el("h2", null, "Все узлы · " + (st.mm === "0" ? "multimask=False" : "multimask=True") + " · DM, % к базе"));
  box.append(el("p", {class: "note", style: "margin:0 0 8px"}, "Для каждого узла группа токенов и α выбраны на COCO A вне фолда и заморожены. В ячейке — bottom50 / top50 / все; ★ — лучший узел по COCO A."));
  const tb = el("table");
  tb.append(el("thead", null, el("tr", null, el("th", null, "Узел"), el("th", null, "Выбор на A"), ...D.datasets.map(d => el("th", {class: "num"}, DSN(d))))));
  const body = el("tbody");
  for (const s of D.sites) {
    const c = chosen(s.id);
    const tr = el("tr", {class: "pick" + (s.id === st.site ? " cur" : "")}, el("td", null, s.id + (s.id === sel().best ? " ★" : "")), el("td", null, TGL[c.group] + ", α " + alphaT(c.alpha)));
    for (const d of D.datasets) {
      const v = ["bottom50", "top50", "all"].map(g => stat(d, s.id, "dm", g));
      tr.append(el("td", {class: "num"}, v.map(x => x ? fp(x.pct, 1) : "—").join(" / ")));
    }
    tr.tabIndex = 0; tr.addEventListener("click", () => select(s.id)); tr.addEventListener("keydown", ev => { if (ev.key === "Enter") select(s.id); });
    body.append(tr);
  }
  tb.append(body);
  box.append(el("div", {class: "tbl"}, tb));
}
function drawMethods() {
  const box = $("methods"); box.replaceChildren();
  const sid = st.site, mm = st.mm, refs = (D.refs || {})[mm] || {};
  box.append(el("h2", null, "Steering в " + sid + " против других методов · " + (mm === "0" ? "база — один выход" : "база — head-select")));
  const names = [...new Set(Object.values(refs).flatMap(r => Object.values(r).flatMap(x => Object.keys(x))))];
  if (!names.length) box.append(el("p", {class: "note"}, "Строк сравнения нет: прогон без --reference_csv."));
  const tb = el("table");
  tb.append(el("thead", null, el("tr", null, el("th", null, "Набор"), el("th", null, "Боксы"), el("th", {class: "num"}, "База IoU"), el("th", {class: "num"}, "Steering"), ...names.map(n => el("th", {class: "num"}, n)))));
  const body = el("tbody");
  for (const d of D.datasets) for (const g of ["all", "bottom50", "top50"]) {
    const v = stat(d, sid, "dm", g); if (!v) continue;
    const rr = (refs[d] || {})[g] || {};
    body.append(el("tr", {class: g === st.g ? "cur" : ""}, el("td", null, g === "all" ? DSN(d) : ""), el("td", null, GL[g]), el("td", {class: "num"}, f3(v.base)), el("td", {class: "num"}, fp(v.pct)),
      ...names.map(n => el("td", {class: "num"}, rr[n] ? fp(rr[n].pct) : "—"))));
  }
  tb.append(body);
  box.append(el("div", {class: "tbl"}, tb));
  box.append(el("p", {class: "note"}, "Все проценты — к базе этого режима на тех же боксах. head-select, best-of-N и grad — из --reference_csv (у grad и best-of-N свой режим выхода, см. grad_results)."));
}

// ---------- header, controls, checks ----------
function drawHeader() {
  const m = D.meta;
  $("h1").textContent = "Steering " + m.model + " на боксах user study: где в декодере работает фиксированный вектор";
  document.title = "Steering scan · " + m.model;
  $("meta").textContent = m.model + " · " + (m.checkpoint || "HF") + " · " + LY.decoder + " · " + D.datasets.map(d => DSN(d) + " (" + (m.images[d] || "?") + " картинок, " + (m.boxes[d] || "?") + " боксов)").join(" · ") +
    " · α ∈ {" + m.alphas.join(", ") + "} · прогон " + m.created + " · отчёт " + m.report_created;
  const chips = $("chips"); chips.replaceChildren();
  chips.append(el("div", {class: "chip"}, "Токены декодера (" + LY.n_tokens + "): ", el("b", null, LY.token_names.join(" · ")), "; сетка картинки " + LY.emb_hw.join(" × ") + ", кадр " + LY.frame));
  for (const mm of ["0", "1"]) {
    const b = D.sel[mm].best, c = D.sel[mm].sites[b];
    const t = tests.map(d => { const v = ((D.stats[mm][d] || {}).sites || {})[b]; return v ? DSN(d) + " " + fp(v.dm.bottom50.pct, 1) : null; }).filter(Boolean);
    chips.append(el("div", {class: "chip"}, (mm === "0" ? "multimask=False" : "multimask=True") + ": лучший узел ", el("b", null, b), " (" + TGL[c.group] + ", α " + c.alpha + "); bottom50 на тестах: ", el("b", null, t.join(", "))));
  }
  const l2 = Object.entries(D.sel["0"].sites).sort((a, b) => b[1].score_A - a[1].score_A).findIndex(([k]) => k === "L2.norm2");
  if (l2 >= 0) chips.append(el("div", {class: "chip"}, "L2.norm2 (узел SAM-1 по critical shifts): ", el("b", null, (l2 + 1) + "-е место"), " из " + D.sites.length + " по COCO A, multimask=False"));
  if (LY.dynamic_multimask_via_stability) {
    const t = D.datasets.map(d => DSN(d) + " " + share(D.base_head["0"][d])).join(", ");
    chips.append(el("div", {class: "chip"}, "Fallback по стабильности (порог " + LY.stability_thresh + "): при multimask=False маска берётся не с token 0 у ", el("b", null, t)));
  }
}
function seg(id, items, cur, onpick) {
  const s = $(id); s.replaceChildren();
  for (const [v, label, color] of items) {
    const b = el("button", {type: "button", "aria-pressed": String(v === cur)});
    if (color) b.append(el("span", {class: "dot", style: "background:" + color}));
    b.append(label); b.addEventListener("click", () => onpick(v)); s.append(b);
  }
}
function drawControls() {
  seg("seg-mm", [["0", "multimask=False"], ["1", "multimask=True"]], st.mm, v => { st.mm = v; if (!SITE[st.site]) st.site = sel().best; render(true); });
  seg("seg-ds", D.datasets.map(d => [d, DSN(d)]), st.ds, v => { st.ds = v; render(); });
  seg("seg-m", ["dm", "rand", "oracle"].map(m => [m, M[m].name, mcol(m)]), st.m, v => { st.m = v; render(); });
  seg("seg-g", [["bottom50", "bottom50"], ["top50", "top50"], ["all", "все"]], st.g, setG);
  seg("seg-c", [["d", "прирост IoU"], ["auc", "AUC пробы"]], st.c, v => { st.c = v; render(); });
}
function drawChecks() {
  const c = D.checks, b = $("checks-body"); b.replaceChildren();
  const li = (cls, text) => el("li", {class: cls}, text);
  const ul = el("ul");
  const z = c.zero_dose || {};
  ul.append(li(z.ok ? "ok" : "fail", "нулевая доза воспроизводит прогон без вектора бит в бит: " + z.bit_exact + "/" + z.forwards));
  const o = c.own_forward || {};
  ul.append(li(o.ok ? "ok" : "fail", "свой проход декодера (predict_masks + выбор головы модели) = MaskDecoder.forward бит в бит: " + o.bit_exact + "/" + o.compared));
  const p = c.public_api || {};
  ul.append(li(p.max_abs_iou_diff <= 1e-4 && !(p.where || []).length ? "ok" : "warn", "публичный predict / predict_torch на тех же боксах: max |ΔIoU| " + (ok(p.max_abs_iou_diff) ? p.max_abs_iou_diff.toExponential(1) : "—") + " (" + p.compared + " боксов)" + (p.where && p.where.length ? "; " + p.where.slice(0, 4).join("; ") : "")));
  const q = c.public_single;
  if (q) ul.append(li(q.max_abs_iou_diff <= 1e-3 && !(q.where || []).length ? "ok" : "warn", "публичный predict по одному боксу (кадр бокса; остаток — зависимость от батча): max |ΔIoU| " + (ok(q.max_abs_iou_diff) ? q.max_abs_iou_diff.toExponential(1) : "—") + ", " + q["n_over_1e-3"] + "/" + q.compared + " боксов > 1e-3" + (q.where && q.where.length ? "; " + q.where.slice(0, 4).join("; ") : "")));
  const tf = c.tf32;
  if (tf) ul.append(li(tf.matmul_run ? "warn" : "ok", "TF32 в матричных умножениях: после загрузки модели " + tf.matmul_after_load + ", в прогоне " + tf.matmul_run + "; cuDNN TF32 " + tf.cudnn));
  for (const [k, lab] of [["repro_A", "COCO A, один выход против reference csv (стоп-проверка)"], ["repro_mm0", "все боксы, один выход против undefended_iou"], ["repro_mm1", "все боксы, head-select против headsel_iou"]]) {
    const r = c[k]; if (!r) continue;
    if (!r.available) { ul.append(li("warn", lab + ": reference csv не задан")); continue; }
    ul.append(li((r.ok != null ? r.ok : r.frac_over_tol <= 0.05) ? "ok" : "warn", lab + ": max |Δ| " + (ok(r.max_abs) ? r.max_abs.toExponential(1) : "—") + ", " + r.n_over_tol + "/" + r.matched + " боксов > " + r.tol + "; среднее " + f3(r.mean_ours) + " против " + f3(r.mean_ref)));
  }
  const br = c.baseline_repeat || {};
  ul.append(li(br.max_abs_iou_diff === 0 ? "ok" : "warn", "COCO A закодирован заново для второго прохода: max |ΔIoU| " + br.max_abs_iou_diff + " (" + br.images + " картинок)"));
  const oc = c.oracle_cut_prompt_tok || {};
  ul.append(li((oc.max_abs_iou_vs_best || 0) <= 1e-3 ? "ok" : "warn", "oracle в prompt tokens воспроизводит лучший бокс (разрез графа): max |ΔIoU| " + oc.max_abs_iou_vs_best));
  const g = c.graph || {};
  ul.append(li(g.ok ? "ok" : "warn", "граф: сработали " + g.fired + " модулей декодера; не нарисованы: " + ((g.undrawn || []).join(", ") || "нет") + "; нарисованы, но не сработали: " + ((g.drawn_not_fired || []).join(", ") || "нет")));
  const gr = c.groups || {};
  ul.append(li(gr.ok ? "ok" : "fail", "группы 12 / 13 на каждой картинке: " + gr.images + " картинок"));
  const sp = c.split || {}, vb = c.vector_boxes || {};
  ul.append(li(sp.ok && vb.ok ? "ok" : "fail", "сплит COCO: A " + sp.A + " / B " + sp.B + ", не пересекаются; вектор усредняет " + vb.used + " боксов A из " + vb.expected));
  if (D.expect) ul.append(li(Object.values(D.expect).every(x => x.ok) ? "ok" : "warn", "SAM-1 повторяет steer_user_study (L2.norm2, все токены, α = 1, Δ всех боксов): " + Object.entries(D.expect).map(([k, x]) => DSN(k) + " " + fd(x.got) + " (было " + fd(x.want) + ")").join(", ")));
  b.append(ul);
  b.append(el("p", {class: "note"}, "Версии: " + Object.entries(D.meta.versions || {}).map(([k, v]) => k + " " + v).join(", ") + " · устройство " + D.meta.device + " · " + D.meta.seconds + " с"));
}

// ---------- state ----------
function setG(g) { st.g = g; render(); }
function select(site) { st.site = site; render(true); }
function writeHash() { history.replaceState(null, "", "#" + new URLSearchParams({mm: st.mm, ds: st.ds, m: st.m, g: st.g, c: st.c, site: st.site})); }
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (["0", "1"].includes(p.get("mm"))) st.mm = p.get("mm");
  st.site = sel().best;
  if (D.datasets.includes(p.get("ds"))) st.ds = p.get("ds");
  if (M[p.get("m")]) st.m = p.get("m");
  if (GL[p.get("g")]) st.g = p.get("g");
  if (["d", "auc"].includes(p.get("c"))) st.c = p.get("c");
  if (SITE[p.get("site")]) st.site = p.get("site");
}
function render(scroll) { hideTip(); drawControls(); drawLegend(); drawGraph(scroll); drawSite(); drawLayers(); drawDose(); drawOverview(); drawMethods(); writeHash(); }
$("btn-terms").addEventListener("click", () => { const t = $("terms"); t.open = true; t.scrollIntoView({behavior: "smooth"}); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render());
let rsz = 0, lastW = 0;
addEventListener("resize", () => { clearTimeout(rsz); rsz = setTimeout(() => { const w = $("ch-dose").clientWidth; if (w !== lastW) { lastW = w; drawLayers(); drawDose(); } }, 150); });
readHash(); drawHeader(); drawChecks(); render(true);
})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
