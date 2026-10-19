"""
steer_report.py
===============
One self-contained HTML page on a steer_sam_decoder.py run.

  * SAM's prompt encoder + mask decoder as the dataflow graph of
    activation_flow_graph.py; the 12 steering sites are clickable, the chosen
    one is highlighted and everything it cannot affect is dimmed;
  * the method: DM, Mean, the random direction of DM's norm (control) or the
    oracle (activation patching from the pair's own best box);
  * per site: restore at the matched change budget with its 95% CI, the
    controls, the harm on good boxes and on boxes around the GT, the dose curve
    (restore vs harm over alpha) and the layer profile with the probe AUC;
  * example pairs: best box, bad box, bad box + vector, best box + vector.

The example masks are RECOMPUTED, not approximated:
  boxes    prepare_image() with the run's seeds -> the run's own boxes;
  vectors  source: out of fold, rebuilt from re-collected features, checked
           against the sha256 in checks.json; transfer: vectors.npz (the run's
           source-wide vectors), checked the same way;
  configs  the run's choice at --budget (on the source, the choice of the fold
           the image was held out in).
Every recomputed IoU is compared with <dataset>_pairs.csv.gz; the largest
deviation is printed and shown in the report.

Example
-------
    CUDA_VISIBLE_DEVICES=0 python scripts/steer_report.py --run_dir exp_res/steer_sam

    # numbers only, no SAM / data needed
    python scripts/steer_report.py --run_dir exp_res/steer_sam --no_examples
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from activation_flow_graph import build_spec  # noqa: E402

PICK_METHODS = ("dm", "mean", "dm_rand", "oracle")           # selectable; example masks are computed for these
TABLE_METHODS = ("dm", "mean", "dm_rand", "dm_rev", "const", "oracle")
CHOSEN_RE = re.compile(r"f(\d+):(\S+)")

GROUP_RU = {"all": "все 7 токенов", "mask0": "mask token 0", "mask123": "mask tokens 1–3",
            "prompt": "prompt-токены", "iou": "IoU token", "box": "клетки внутри бокса", "-": ""}
SHORT_LABEL = {"tokens": "tokens: output + prompt", "src": "image emb. + 'no mask'",
               "corner_pe": "corner PE", "corner_emb": "corner embeddings", "img_pe": "image PE",
               "hyper": "hypernet MLP", "low_res": "mask logits 256²", "mask_logits": "upscaled logits",
               "img_emb": "image embedding"}


# ---------------------------------------------------------------------------
# Run outputs
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    checks = json.loads((run_dir / "checks.json").read_text())
    names = [n for n in [meta["source"], *meta.get("transfer", [])] if (run_dir / f"{n}_calibrated.csv").is_file()]
    img_dt = {"image": str}
    return {
        "dir": run_dir, "meta": meta, "checks": checks, "names": names,
        "cfg": pd.read_csv(run_dir / "configs.csv"),
        "oracle": pd.read_csv(run_dir / "oracle.csv"),
        "probe": pd.read_csv(run_dir / "probe_layers.csv"),
        "cal": {n: pd.read_csv(run_dir / f"{n}_calibrated.csv") for n in names},
        "grid": {n: pd.read_csv(run_dir / f"{n}_grid.csv") for n in names},
        "pairs": {n: pd.read_csv(run_dir / f"{n}_pairs.csv.gz", dtype=img_dt) for n in names},
        "images": {n: pd.read_csv(run_dir / f"{n}_images.csv", dtype=img_dt) for n in names},
    }


def depth_of(cfg: pd.DataFrame) -> int:
    layers = {int(m.group(1)) for s in cfg.site.unique() if (m := re.match(r"L(\d+)\.", s))}
    return max(layers) if layers else 0


def parse_choice(s: str):
    """'prompt/a=0.25' -> ('prompt', 0.25); '-' -> None."""
    if not isinstance(s, str) or s == "-" or "/a=" not in s:
        return None
    g, a = s.rsplit("/a=", 1)
    return g, float(a)


def human_alpha(a: float) -> str:
    if a < 1 and abs(1 / a - round(1 / a)) < 1e-9 and round(1 / a) & (round(1 / a) - 1) == 0:
        return f"1/{round(1 / a)}"
    return f"{a:g}"


def human_choice(choice) -> str:
    if choice is None:
        return ""
    g, a = choice
    grp = GROUP_RU.get(g, g)
    return (grp + ", " if grp else "") + "α = " + human_alpha(a)


class Picks:
    """The run's config choice at a budget: (dataset, site, method, fold) -> cid.
    On the source every fold has its own choice (made on the other folds); on a
    transfer dataset one frozen choice (fold -1)."""

    def __init__(self, R: dict, budget: float):
        cfg = R["cfg"]
        self.cfg = cfg.set_index("cid")
        self.src = R["names"][0]
        steer = cfg[cfg.kind == "steer"]
        self._idx = {(r.site, r.method, r.group, float(r.alpha)): int(r.cid) for r in steer.itertuples()}
        self.oracle = {r.site: int(r.cid) for r in cfg[(cfg.kind == "oracle") & cfg.whole].itertuples()}
        self.map, self.human = {}, {}
        for ds, cal in R["cal"].items():
            for r in cal[np.isclose(cal.budget, budget)].itertuples():
                if ds == self.src:
                    ch = {int(k): parse_choice(v) for k, v in CHOSEN_RE.findall(str(r.chosen))}
                    self.map[(ds, r.site, r.method)] = {k: self._lookup(r.site, r.method, c) for k, c in ch.items()}
                    cnt = Counter(human_choice(c) or "нет конфигурации в бюджете" for c in ch.values())
                    self.human[(ds, r.site, r.method)] = (
                        next(iter(cnt)) if len(cnt) == 1 else
                        "по фолдам: " + "; ".join(f"{t} ×{n}" for t, n in cnt.most_common()))
                else:
                    cid = None if pd.isna(r.cid) else int(r.cid)
                    self.map[(ds, r.site, r.method)] = {-1: cid}
                    self.human[(ds, r.site, r.method)] = human_choice(parse_choice(r.chosen)) or \
                        "нет конфигурации в бюджете"

    def _lookup(self, site, method, choice):
        if choice is None:
            return None
        g, a = choice
        for (s, m, gg, aa), cid in self._idx.items():
            if s == site and m == method and gg == g and math.isclose(aa, a, rel_tol=1e-9):
                return cid
        raise KeyError(f"config not in configs.csv: {site}/{method}/{g}/a={a}")

    def cid(self, ds, site, method, fold):
        if method == "oracle":
            return self.oracle.get(site)
        d = self.map.get((ds, site, method), {})
        return d.get(fold if ds == self.src else -1)

    def mode_group(self, ds, site, method):
        """The group most folds chose (for the dose curve)."""
        cids = [c for c in self.map.get((ds, site, method), {}).values() if c is not None]
        if not cids:
            return None
        return Counter(self.cfg.loc[c, "group"] for c in cids).most_common(1)[0][0]


def fold_of(R: dict, ds: str) -> dict:
    if ds != R["names"][0]:
        return {}
    I = R["images"][ds]
    return dict(I.drop_duplicates("image")[["image", "fold"]].itertuples(index=False))


def pair_restore(R: dict, picks: Picks, ds: str, site: str, method: str) -> pd.DataFrame:
    """Per pair: restore of the config the run applied to it."""
    P = R["pairs"][ds]
    fo = fold_of(R, ds)
    rows = []
    for (image, pid), g in P.groupby(["image", "pid"], sort=False):
        cid = picks.cid(ds, site, method, fo.get(image, -1))
        r = g[g.cid == cid]
        if cid is not None and len(r):
            rows.append({"image": image, "pid": int(pid), "restore": float(r.restore.iloc[0]),
                         "iou_bad0": float(r.iou_bad0.iloc[0]), "iou_best0": float(r.iou_best0.iloc[0])})
    return pd.DataFrame(rows, columns=["image", "pid", "restore", "iou_bad0", "iou_best0"])


def choose_examples(R, picks, ds, site, k, forced=()) -> list[int]:
    """k pairs on distinct images, spread over the quantiles of the headline
    config's restore (so failures are shown too); forced pids first. Without a
    DM config in the budget the oracle's restore orders them instead."""
    pr = pair_restore(R, picks, ds, site, "dm").dropna(subset=["restore"])
    if pr.empty:
        pr = pair_restore(R, picks, ds, site, "oracle").dropna(subset=["restore"])
    out = [p for p in forced if p in set(pr.pid)]
    used = set(pr[pr.pid.isin(out)].image)
    pr = pr.sort_values("restore").reset_index(drop=True)
    for i in range(max(0, k - len(out))):
        target = (i + 0.5) / max(1, k - len(out)) * (len(pr) - 1)
        for j in np.argsort(np.abs(np.arange(len(pr)) - target), kind="stable"):
            if pr.image[j] not in used:
                out.append(int(pr.pid[j]))
                used.add(pr.image[j])
                break
    return out


# ---------------------------------------------------------------------------
# Payload: numbers
# ---------------------------------------------------------------------------

def _f(x, nd=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, nd)


def site_info(site) -> dict:
    sid = site.id
    m = re.match(r"L(\d+)\.norm(\d)", sid)
    kind = "action" if site.kind in ("vec", "logits") else "residual"
    if sid == "prompt_tok":
        title, text = "prompt tokens", (
            "Два токена углов бокса на выходе prompt encoder (sparse_embeddings). Вектор прибавляется к обоим; "
            "через «re-added prompt tokens» он попадает и в каждую attention декодера.")
    elif m:
        li, k = int(m.group(1)), int(m.group(2))
        title = {1: f"слой {li}: после self-attention", 2: f"слой {li}: после token → image attention",
                 3: f"слой {li}: после MLP", 4: f"слой {li}: image embedding после image → token attention"}[k]
        text = {
            1: "Токены декодера (IoU, 4 mask, 2 prompt) после self-attention и LayerNorm norm1"
               + (": в слое 1 без residual, self-attention заменяет токены." if li == 1 else "."),
            2: "Токены после того, как они прочитали картинку (token → image attention), LayerNorm norm2.",
            3: "Токены после поточечного MLP, LayerNorm norm3.",
            4: "Image embedding (64 × 64 клетки × 256) после того, как картинка прочитала токены, LayerNorm norm4. "
               "Один 256-мерный вектор прибавляется к клеткам, центр которых лежит внутри бокса.",
        }[k]
    elif sid == "final_norm":
        title, text = "после финальной token → image attention", (
            "Последнее состояние токенов. На маску влияет только mask token 0 (multimask_output=False), "
            "поэтому вектор прибавляется к нему.")
    elif sid == "hyper":
        title, text = "выход гиперсети", (
            "32 веса динамического классификатора: логит пикселя = ⟨веса, upscaled embedding(x, y)⟩. "
            "Сдвиг прямо на интерфейсе решения, как action-head steering в статье.")
    else:
        title, text = "логиты маски 256 × 256", (
            "Вектор — поле логитов G × G в кадре бокса (бокс + 25% с каждой стороны), растянутое на каждый бокс. "
            "Каждый пиксель — бинарное решение, как gripper у Magma. Контроль «сдвиг порога» прибавляет "
            "одну константу ко всем пикселям.")
    return {"id": sid, "kind": site.kind, "module": site.module, "title": title, "text": text, "type": kind}


def probe_group(kind: str) -> str:
    return {"sparse": "prompt", "tok": "all", "img": "box"}.get(kind, "-")


def numbers_payload(R: dict, picks: Picks, budget: float, sites) -> dict:
    names = R["names"]
    src = names[0]
    cal, orc = {}, {}
    for ds in names:
        c = R["cal"][ds]
        c = c[np.isclose(c.budget, budget)]
        cal[ds] = {}
        for r in c.itertuples():
            cal[ds].setdefault(r.site, {})[r.method] = {
                "restore": _f(r.restore), "lo": _f(r.restore_lo), "hi": _f(r.restore_hi), "fixed": _f(r.fixed),
                "chg_good": _f(r.chg_good), "d_good": _f(r.d_good), "d_dense": _f(r.d_dense),
                "lt05": _f(r.lt05_delta), "cfg": picks.human.get((ds, r.site, r.method), "")}
        o = R["oracle"]
        o = o[(o.dataset == ds) & o.whole]
        orc[ds] = {r.site: {"restore": _f(r.restore), "lo": _f(r.restore_lo), "hi": _f(r.restore_hi),
                            "fixed": _f(r.fixed), "chg_good": 0.0, "d_good": 0.0, "d_dense": 0.0, "lt05": 0.0,
                            "cfg": "тензор из best той же пары"} for r in o.itertuples()}
    P = R["probe"].set_index(["site", "group"])
    probe = {}
    for s in sites:
        key = (s.id, probe_group(s.kind))
        if key not in P.index:
            continue
        row = P.loc[key]
        probe[s.id] = {"auc": {ds: _f(row.get("auc_vs_ctrl" if ds == src else f"auc_vs_ctrl@{ds}")) for ds in names},
                       "off_bad": {ds: _f(row.get("off_bad" if ds == src else f"off_bad@{ds}")) for ds in names}}
    G = R["grid"][src]
    G = G[G.kind == "steer"]
    dose = {}
    for s in sites:
        dose[s.id] = {}
        g_dm = picks.mode_group(src, s.id, "dm")
        for m in ("dm", "mean", "dm_rand", "dm_rev", "const"):
            g = picks.mode_group(src, s.id, "mean") if m == "mean" else g_dm
            sub = G[(G.site == s.id) & (G.method == m)]
            if g is not None:
                sub = sub[sub.group == g]
            elif len(sub):
                sub = sub[sub.group == sub.group.iloc[0]]
            if not len(sub):
                continue
            sub = sub.sort_values("alpha")
            dose[s.id][m] = {"group": GROUP_RU.get(sub.group.iloc[0], sub.group.iloc[0]),
                             "pts": [[_f(a, 8), _f(c, 5), _f(r), _f(lo), _f(hi)] for a, c, r, lo, hi in
                                     zip(sub.alpha, sub.chg_good, sub.restore, sub.restore_lo, sub.restore_hi)]}
    return {"cal": cal, "oracle": orc, "probe": probe, "dose": dose}


def findings(R, num, sites, src) -> dict:
    ids = [s.id for s in sites]
    dm = {s: (num["cal"][src].get(s, {}).get("dm") or {}).get("restore") for s in ids}
    best = max((s for s in ids if dm[s] is not None), key=lambda s: dm[s], default=ids[0])
    first_auc = next((s for s in ids if ((num["probe"].get(s) or {}).get("auc") or {}).get(src) is not None
                      and num["probe"][s]["auc"][src] >= 0.65), None)
    return {"site": best, "first_auc": first_auc}


# ---------------------------------------------------------------------------
# Payload: example masks (needs SAM and the data)
# ---------------------------------------------------------------------------

def _b64(buf) -> str:
    return base64.b64encode(buf.tobytes()).decode("ascii")


_BILEVEL = [getattr(cv2, "IMWRITE_PNG_BILEVEL", 18), 1]


class Thumb:
    """Crop around the pair's two boxes (original pixels), scaled to <= h x 1.6h."""

    def __init__(self, boxes_o: np.ndarray, H: int, W: int, h: int):
        x0, y0 = boxes_o[:, 0].min(), boxes_o[:, 1].min()
        x1, y1 = boxes_o[:, 2].max(), boxes_o[:, 3].max()
        mx, my = 0.15 * (x1 - x0) + 10, 0.15 * (y1 - y0) + 10
        self.X0, self.Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        self.X1, self.Y1 = int(min(W, x1 + mx + 1)), int(min(H, y1 + my + 1))
        cw, ch = self.X1 - self.X0, self.Y1 - self.Y0
        self.s = min(h / max(1, ch), 1.6 * h / max(1, cw))
        self.w, self.h = max(1, round(cw * self.s)), max(1, round(ch * self.s))

    def crop(self, a: np.ndarray) -> np.ndarray:
        return a[self.Y0:self.Y1, self.X0:self.X1]

    def box(self, b) -> list:
        return [round((b[0] - self.X0) * self.s, 1), round((b[1] - self.Y0) * self.s, 1),
                round((b[2] - self.X0) * self.s, 1), round((b[3] - self.Y0) * self.s, 1)]

    def base(self, img_rgb: np.ndarray, gt: np.ndarray) -> str:
        """JPEG of the crop with the GT outline (green, dark halo)."""
        t = cv2.resize(cv2.cvtColor(self.crop(img_rgb), cv2.COLOR_RGB2BGR), (self.w, self.h),
                       interpolation=cv2.INTER_AREA)
        g = cv2.resize(self.crop(gt).astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, (0, 0, 0), 3)
        cv2.drawContours(t, cs, -1, (60, 220, 60), 1)
        ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return _b64(buf)

    def mask(self, m: np.ndarray) -> str:
        r = cv2.resize(self.crop(m).astype(np.uint8) * 255, (self.w, self.h), interpolation=cv2.INTER_AREA) > 127
        ok, buf = cv2.imencode(".png", r.astype(np.uint8) * 255, _BILEVEL)
        return _b64(buf)


def compute_examples(R: dict, picks: Picks, sel: dict, args) -> tuple[dict, dict]:
    import torch
    from activation_flow_graph import load_gt, load_image_rgb, load_predictor
    from steer_sam_decoder import (Cfg, Runner, assign_folds, batch_iou, build_sites, build_vectors, collect,
                                   donor_recorder, make_edit, prepare_image, random_directions, resolve_datasets,
                                   vectors_sha256, vectors_to_device)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ra = argparse.Namespace(**R["meta"]["args"])
    for k in ("checkpoint_path", "images_dir", "masks_dir", "datasets_root", "device"):
        if getattr(args, k) is not None:
            setattr(ra, k, getattr(args, k))
    if Path(R["meta"].get("source_json", "")).is_file():
        ra.critical_shifts = R["meta"]["source_json"]
    device = torch.device(ra.device)
    src, targets, _ = resolve_datasets(ra)
    predictor = load_predictor(ra.checkpoint_path, ra.model_type, device)
    sites = build_sites(len(predictor.model.mask_decoder.transformer.layers))
    runner = Runner(predictor, sites, ra.batch, ra.field_grid, ra.field_margin)
    sink = {"repro": [], "base": [], "base_dense": []}
    rc = {"sha_fold_ok": None, "sha_all_ok": None, "fold_assign_ok": None, "iou_checked": 0,
          "iou_max_dev": 0.0, "iou_max_where": "", "missing_rows": 0}
    want_sha = R["checks"].get("vectors_sha256", {})

    # source: re-collect every image for the out-of-fold vectors
    items, feats, fmeta = collect(runner, src, ra, sink)
    folds = assign_folds([it["image"] for it in items], min(ra.folds, len(items)), ra.seed)
    fo_run = fold_of(R, src.name)
    rc["fold_assign_ok"] = all(fo_run.get(k) == v for k, v in folds.items())
    fold_row = fmeta.image.map(folds).to_numpy()
    good = (fmeta.role != "bad").to_numpy()
    rnd = random_directions(sites, feats, ra.seed)
    V_fold = {k: build_vectors(feats, good & (fold_row != k), ~good & (fold_row != k), sites, rnd)
              for k in sorted(set(folds.values()))}
    rc["sha_fold_ok"] = all(vectors_sha256(v) == want_sha.get(f"fold{k}") for k, v in V_fold.items())
    npz = np.load(R["dir"] / "vectors.npz")
    V_all = {}
    for key in npz.files:
        s, m = key.split("/", 1)
        v = npz[key]
        V_all.setdefault(s, {})[m] = float(v) if v.ndim == 0 else v
    rc["sha_all_ok"] = vectors_sha256(V_all) == want_sha.get("all")
    for k, ok in (("sha_fold_ok", rc["sha_fold_ok"]), ("sha_all_ok", rc["sha_all_ok"]),
                  ("fold_assign_ok", rc["fold_assign_ok"])):
        if not ok:
            print(f"[warn] {k} = False: the re-collected source differs from the run "
                  f"(other GPU / library?) -- the IoU check below shows by how much", flush=True)
    Vt_fold = {k: vectors_to_device(v, device) for k, v in V_fold.items()}
    Vt_all = vectors_to_device(V_all, device)
    cfg_i = R["cfg"].set_index("cid")
    out = {}

    def render(ds_name, item, gt, pids, Vt, fold):
        boxes = torch.as_tensor(item["boxes"], device=device)
        role = torch.as_tensor(item["role"], device=device)
        store = {s.id: [] for s in sites}
        m0, _ = runner.forward(boxes, record={s.id: donor_recorder(store, s.id, role == 0) for s in sites})
        donors = {k: torch.cat(v) for k, v in store.items()}
        donor_of = torch.full((len(boxes),), -1, dtype=torch.long, device=device)
        donor_of[torch.as_tensor(item["ri"], device=device)] = torch.arange(len(item["ri"]), device=device)
        iou0 = batch_iou(m0, gt).cpu().numpy()
        img = load_image_rgb(Path(item["img_path"]))
        gt_np = gt.cpu().numpy()
        H, W = gt_np.shape
        in_h, in_w = predictor.input_size
        sc = np.array([in_w / W, in_h / H, in_w / W, in_h / H])
        P = R["pairs"][ds_name]
        exs = {}
        for pid in pids:
            k = item["pids"].index(pid)
            bi, ri = int(item["bi"][k]), int(item["ri"][k])
            b_o, r_o = item["boxes"][bi] / sc, item["boxes"][ri] / sc
            th = Thumb(np.stack([b_o, r_o]), H, W, args.thumb_h)
            mb, mr = m0[bi].cpu().numpy(), m0[ri].cpu().numpy()
            exs[pid] = {"ds": ds_name, "image": item["image"], "pid": int(pid), "fold": int(fold),
                        "shift": round(float(np.abs(b_o - r_o).max()), 1), "w": th.w, "h": th.h,
                        "base": th.base(img, gt_np), "box_best": th.box(b_o), "box_bad": th.box(r_o),
                        "iou_best0": _f(iou0[bi]), "iou_bad0": _f(iou0[ri]),
                        "m_best0": th.mask(mb), "m_bad0": th.mask(mr), "steer": {}, "_th": th, "_mb": mb,
                        "_bi": bi, "_ri": ri}
        for s in sites:
            for method in PICK_METHODS:
                cid = picks.cid(ds_name, s.id, method, fold)
                if cid is None:
                    for e in exs.values():
                        e["steer"].setdefault(s.id, {})[method] = None
                    continue
                row = cfg_i.loc[cid]
                cfg = Cfg(int(cid), row.kind, row.site, row.group, row.method, float(row.alpha))
                m, _ = runner.forward(boxes, edits={s.id: make_edit(cfg, runner.sites[s.id], Vt, donors, donor_of)})
                iou = batch_iou(m, gt).cpu().numpy()
                for pid, e in exs.items():
                    bi, ri = e["_bi"], e["_ri"]
                    r = P[(P.pid == pid) & (P.cid == cid)]
                    if len(r):
                        dev = max(abs(float(r.iou_bad.iloc[0]) - float(iou[ri])),
                                  abs(float(r.iou_best.iloc[0]) - float(iou[bi])))
                        rc["iou_checked"] += 2
                        if dev > rc["iou_max_dev"]:
                            rc["iou_max_dev"], rc["iou_max_where"] = dev, f"{ds_name} pair {pid} {s.id}/{method}"
                    else:
                        rc["missing_rows"] += 1
                    mb_new = m[bi].cpu().numpy()
                    same = bool(np.array_equal(mb_new, e["_mb"]))
                    den = e["iou_best0"] - e["iou_bad0"]
                    e["steer"].setdefault(s.id, {})[method] = {
                        "cfg": "тензор из best той же пары" if method == "oracle" else
                        human_choice((row.group, float(row.alpha))),
                        "iou_bad": _f(iou[ri]), "iou_best": _f(iou[bi]),
                        "restore": _f((float(iou[ri]) - e["iou_bad0"]) / den) if den > 1e-6 else None,
                        "m_bad": e["_th"].mask(m[ri].cpu().numpy()),
                        "m_best": None if same else e["_th"].mask(mb_new)}
        for e in exs.values():
            for k in ("_th", "_mb", "_bi", "_ri"):
                e.pop(k)
        return [exs[p] for p in pids]

    t0 = time.time()
    want = set(sel.get(src.name, []))
    rows = []
    for it in items:
        pids = [p for p in it["pids"] if p in want]
        if not pids:
            continue
        with torch.inference_mode():
            predictor.set_image(load_image_rgb(Path(it["img_path"])))
        gt = torch.from_numpy(load_gt(Path(it["gt_path"]))).to(device)
        rows += render(src.name, it, gt, pids, Vt_fold[folds[it["image"]]], folds[it["image"]])
        print(f"[{src.name}] examples: {it['image']} ({time.time() - t0:.0f}s)", flush=True)
    out[src.name] = sorted(rows, key=lambda e: sel[src.name].index(e["pid"]))

    for ds in targets:
        if ds.name not in sel:
            continue
        cases = json.loads(ds.json.read_text())
        by_img: OrderedDict[str, list] = OrderedDict()
        for pid, c in enumerate(cases):
            by_img.setdefault(c["image_name"], []).append((pid, c))
        want = set(sel[ds.name])
        rows = []
        for name, cs in by_img.items():
            pids = [p for p, _ in cs if p in want]
            if not pids:
                continue
            res = prepare_image(runner, ds, name, cs, ra, sink)
            if res is None:
                print(f"[warn] {ds.name}/{name}: no pair kept on re-prediction", flush=True)
                continue
            item, _, _, gt = res
            rows += render(ds.name, item, gt, [p for p in pids if p in item["pids"]], Vt_all, -1)
            print(f"[{ds.name}] examples: {name} ({time.time() - t0:.0f}s)", flush=True)
        out[ds.name] = sorted(rows, key=lambda e: sel[ds.name].index(e["pid"]))
    return out, rc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(args) -> int:
    run_dir = Path(args.run_dir)
    R = load_run(run_dir)
    meta, src = R["meta"], R["names"][0]
    budgets = meta["args"].get("budgets", [args.budget])
    if not any(math.isclose(b, args.budget) for b in budgets):
        raise SystemExit(f"--budget {args.budget} is not one of the run's budgets {budgets}")
    picks = Picks(R, args.budget)

    from steer_sam_decoder import build_sites
    depth = depth_of(R["cfg"])
    sites = build_sites(depth)
    spec = build_spec(depth)
    spec_out = {
        "lanes": spec["lanes"], "groups": spec["groups"],
        "nodes": [{"id": n["id"], "label": SHORT_LABEL.get(n["id"], n["label"]), "lane": n["lane"], "row": n["row"],
                   "kind": n["kind"], "code": n["code"]} for n in spec["nodes"]],
        "edges": [{"id": e["id"], "src": e["src"], "dst": e["dst"], "role": e["role"]} for e in spec["edges"]],
    }
    num = numbers_payload(R, picks, args.budget, sites)
    head = findings(R, num, sites, src)

    forced = {}
    for tok in args.pids:
        ds, pid = tok.split(":")
        forced.setdefault(ds, []).append(int(pid))
    sel = {ds: choose_examples(R, picks, ds, head["site"], args.examples, forced.get(ds, ()))
           for ds in R["names"]} if args.examples > 0 else {}
    examples, rc = {}, None
    if not args.no_examples and sel:
        examples, rc = compute_examples(R, picks, sel, args)
        print(f"example IoU vs run: {rc['iou_checked']} values, max |diff| {rc['iou_max_dev']:.3g}"
              f"{' at ' + rc['iou_max_where'] if rc['iou_max_where'] else ''}; vectors sha256 "
              f"fold {rc['sha_fold_ok']}, all {rc['sha_all_ok']}; folds {rc['fold_assign_ok']}", flush=True)
        if rc["iou_max_dev"] > args.iou_tol:
            print(f"[warn] recomputed masks differ from the run by up to {rc['iou_max_dev']:.3g} IoU "
                  f"(> --iou_tol {args.iou_tol}); the report flags it", flush=True)

    ck = R["checks"]
    run_checks = {
        "zero_dose": f"{ck['zero_dose']['bit_exact']}/{ck['zero_dose']['forwards']}",
        "trace": f"{ck['trace_vs_hook']['targets']} целей, max rel {ck['trace_vs_hook']['max_rel_low_res']:.3g}",
        "noop": ck["noop"]["max_rel"], "baseline": ck["baseline_repeat"]["max_abs_iou_diff"],
        "oracle_cuts": ck.get("oracle_cuts", {}), "oracle_other": ck.get("oracle_other_boxes_max_change"),
        "repro": ck.get("repro", {}),
    }
    payload = {
        "meta": {"model": meta["model"], "checkpoint": meta["checkpoint"], "source": src,
                 "created": meta["created"], "device": meta.get("device"), "budget": args.budget,
                 "report_created": time.strftime("%Y-%m-%d %H:%M"), "folds": meta.get("folds"),
                 "images": meta.get("images", {}), "pairs": meta.get("pairs", {}),
                 "examples_computed": bool(examples), "iou_tol": args.iou_tol},
        "datasets": [{"name": n, "role": "source" if n == src else "transfer"} for n in R["names"]],
        "spec": spec_out, "sites": [site_info(s) for s in sites], "headline": head,
        **num, "examples": examples, "run_checks": run_checks, "report_checks": rc,
    }
    out = Path(args.out) if args.out else run_dir / "steer_report.html"
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    out.write_text(HTML.replace("__PAYLOAD__", data), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="out_dir of steer_sam_decoder.py")
    p.add_argument("--out", default=None, help="default: <run_dir>/steer_report.html")
    p.add_argument("--budget", type=float, default=0.02, help="change budget on good boxes (one of the run's)")
    p.add_argument("--examples", type=int, default=8, help="example pairs per dataset")
    p.add_argument("--pids", nargs="*", default=[], help="extra example pairs as DATASET:PID")
    p.add_argument("--thumb_h", type=int, default=160)
    p.add_argument("--no_examples", action="store_true", help="numbers only; no SAM, no data")
    p.add_argument("--iou_tol", type=float, default=1e-4, help="allowed |IoU recomputed - IoU in the run|")
    p.add_argument("--checkpoint_path", default=None, help="default: from the run's meta.json")
    p.add_argument("--images_dir", default=None, help="source images; default: from meta.json")
    p.add_argument("--masks_dir", default=None)
    p.add_argument("--datasets_root", nargs="*", default=None)
    p.add_argument("--device", default=None)
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
<title>SAM steering report</title>
<style>
:root{
  --scheme:light;
  --bg:#f9f9f7; --surface:#fcfcfb; --surface-2:#f0efec; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --sel:#0b0b0b; --link:#1c5cab;
  --ok:#006300; --warn:#b5520f; --fail:#d03b3b;
  --m-dm:#2a78d6; --m-mean:#eb6834; --m-rand:#1baf7a; --m-rev:#eda100; --m-const:#e87ba4; --m-oracle:#52514e;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --scheme:dark;
    --bg:#0d0d0d; --surface:#1a1a19; --surface-2:#262624; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#4a4a46; --border:rgba(255,255,255,0.10); --sel:#ffffff; --link:#86b6ef;
    --ok:#0ca30c; --warn:#ec835a; --fail:#e66767;
    --m-dm:#3987e5; --m-mean:#d95926; --m-rand:#199e70; --m-rev:#c98500; --m-const:#d55181; --m-oracle:#c3c2b7;
  }
}
:root[data-theme="dark"]{
  --scheme:dark;
  --bg:#0d0d0d; --surface:#1a1a19; --surface-2:#262624; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#4a4a46; --border:rgba(255,255,255,0.10); --sel:#ffffff; --link:#86b6ef;
  --ok:#0ca30c; --warn:#ec835a; --fail:#e66767;
  --m-dm:#3987e5; --m-mean:#d95926; --m-rand:#199e70; --m-rev:#c98500; --m-const:#d55181; --m-oracle:#c3c2b7;
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
.chip{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:6px 10px;font-size:13px;max-width:520px}
.chip b{font-weight:650}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;padding:8px 0;border-bottom:1px solid var(--grid);margin-bottom:12px}
.seg{display:inline-flex;border:1px solid var(--axis);border-radius:8px;overflow:hidden}
.seg button{font:inherit;font-size:13px;background:var(--surface);color:var(--ink);border:0;border-left:1px solid var(--axis);padding:5px 10px;cursor:pointer;display:inline-flex;align-items:center;gap:6px}
.seg button:first-child{border-left:0}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
.seg button:disabled{color:var(--muted);cursor:not-allowed}
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
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(460px,100%),1fr));gap:18px}
.chart{min-width:0}
.chart svg{display:block;max-width:100%;overflow:visible}
.ckey{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink-2);margin:2px 0 4px}
.ckey span{display:inline-flex;align-items:center;gap:5px}
.ckey i{display:inline-block;width:16px;height:0;border-top:2px solid}
.note{color:var(--ink-2);font-size:12px;margin:4px 0 0}
.ex-head{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline;margin:0 0 8px}
.exs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(700px,100%),1fr));gap:12px;align-items:start}
.ex{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px 10px}
.ex .t{font-size:12.5px;color:var(--ink-2);margin:0 0 6px}
.ex .t b{color:var(--ink)}
.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}
@media (max-width:640px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}
.pn canvas{display:block;width:100%;height:auto;border-radius:4px;background:var(--surface-2)}
.pn .c{font-size:11.5px;color:var(--ink-2);margin-top:3px;line-height:1.3}
.pn .c b{color:var(--ink);font-variant-numeric:tabular-nums}
.pn .na{display:flex;align-items:center;justify-content:center;text-align:center;font-size:11.5px;color:var(--muted);background:var(--surface-2);border-radius:4px;padding:6px}
.up{color:var(--ok)} .down{color:var(--fail)}
.swatch{display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px}
details{margin:14px 0;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px 12px}
summary{cursor:pointer;font-weight:600}
.gloss{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(330px,100%),1fr));gap:10px;margin:10px 0}
.term{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px 12px}
.term h3{margin:0 0 4px}
.term p{margin:4px 0}
.formula{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;background:var(--surface-2);border-radius:6px;padding:5px 8px;margin:5px 0;white-space:pre-wrap}
.tip{position:fixed;z-index:20;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:7px 10px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.18);max-width:360px}
.tip .v{font-variant-numeric:tabular-nums}
.tip .l{color:var(--ink-2)}
.linkbtn{background:none;border:0;padding:0;color:var(--link);font:inherit;font-size:13px;cursor:pointer;text-decoration:underline;text-underline-offset:2px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Steering SAM: где один фиксированный вектор чинит критический сдвиг бокса</h1>
  <p class="meta" id="meta"></p>
  <div class="chips" id="chips"></div>
  <div class="bar" role="toolbar" aria-label="Режим отчёта">
    <div><span class="lbl">Данные</span><span class="seg" id="seg-ds"></span></div>
    <div><span class="lbl">Метод</span><span class="seg" id="seg-m"></span></div>
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
  <section id="examples" style="margin-top:6px"></section>
  <details id="terms">
    <summary>Термины</summary>
    <div class="gloss">
      <div class="term"><h3>Пара best / bad</h3><p>Два бокса на одной картинке в нескольких пикселях друг от друга (critical shift): у best маска хорошая, у bad сломана. <b>Хорошие боксы</b> — best и контроли: стороны best сдвинуты на те же величины, что у bad, но в другие стороны, и маска осталась хорошей.</p></div>
      <div class="term"><h3>Восстановление</h3><div class="formula">(IoU_bad+v − IoU_bad) / (IoU_best − IoU_bad)</div><p>Какую долю потерянного IoU вернул вектор на bad-боксе: 0 — ничего, 100% — маска как у best. Та же формула, что у activation patching.</p></div>
      <div class="term"><h3>Бюджет вреда</h3><p>Вектор прибавляется ко <b>всем</b> боксам, и хорошие маски тоже меняются. Для каждого узла и метода берутся группа токенов и сила α с лучшим восстановлением среди тех, где хорошие маски меняются не больше чем на бюджет (доля изменённых пикселей = 1 − IoU(маска с вектором, маска без)). Так методы сравниваются при одинаковом вреде, а не при одинаковом α, как в GuideVLA.</p></div>
      <div class="term"><h3>DM и Mean</h3><div class="formula">DM:   v = среднее(хорошие) − среднее(bad)
Mean: v = среднее(хорошие)
h' = h + α·v</div><p>Два вида residual steering из GuideVLA. Средние берутся по состояниям в этом узле на других картинках (вне фолда).</p></div>
      <div class="term"><h3>Контроли</h3><p><b>Случайное направление</b> — вектор той же нормы, что DM (для токенов — та же норма на каждом токене), в случайном направлении. Если он чинит так же, дело не в направлении, а в любом толчке. <b>Обратное</b> — −DM. <b>Сдвиг порога</b> (только логиты) — одна константа на все пиксели.</p></div>
      <div class="term"><h3>Oracle (patching)</h3><p>В bad-прогоне тензор узла целиком заменяется тензором best-бокса той же пары. Это ответ, подсмотренный для конкретной пары: потолок того, что можно исправить в этом узле. Фиксированный вектор может чинить и сильнее oracle, если действует через дальнейшие слои по-другому.</p></div>
      <div class="term"><h3>AUC пробы</h3><p>Логистическая регрессия на состояниях узла отличает bad от контролей (сдвиги одинакового размера), кросс-валидация по картинкам. 0.5 — неразличимы, 1 — различимы идеально. На наборах переноса — проба, обученная на источнике.</p></div>
      <div class="term"><h3>Вне фолда и перенос</h3><p><b>Источник</b>: картинки разбиты на фолды; вектор и выбор α для картинки считаются без её фолда. <b>Перенос</b>: векторы и выбор α посчитаны на всём источнике, заморожены и применены к другому набору без подстройки.</p></div>
      <div class="term"><h3>Боксы у GT</h3><p>32 бокса на картинку: GT-бокс со сторонами, случайно сдвинутыми до 10% его размера — боксы, которые нарисовал бы человек. ΔIoU на них показывает, вредит ли вектор обычному использованию.</p></div>
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
const S = D.spec;
const NODE = Object.fromEntries(S.nodes.map(n => [n.id, n]));
const SITE = Object.fromEntries(D.sites.map(s => [s.id, s]));
const SVGNS = "http://www.w3.org/2000/svg";
const LANE_W = 168, NODE_W = 152, NODE_H = 40, ROW_H = 50, GROUP_GAP = 30, PAD = 10, HEAD = 22;
const M = {
  dm:      {name: "DM", long: "разность средних", cv: "--m-dm"},
  mean:    {name: "Mean", long: "среднее хороших", cv: "--m-mean"},
  dm_rand: {name: "Случайное", long: "контроль: случайное направление нормы DM", cv: "--m-rand"},
  dm_rev:  {name: "Обратное", long: "контроль: −DM", cv: "--m-rev"},
  const:   {name: "Сдвиг порога", long: "контроль: константа на все пиксели", cv: "--m-const"},
  oracle:  {name: "Oracle", long: "patching: тензор из best той же пары", cv: "--m-oracle"},
};
const PICK = ["dm", "mean", "dm_rand", "oracle"];
const DSN = {COCO_MVal: "COCO-MVal", Berkeley: "Berkeley", GrabCut: "GrabCut"};
const st = {ds: D.datasets[0].name, m: "dm", c: "restore", site: D.headline.site};
const $ = id => document.getElementById(id);

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else if (k === "style") e.style.cssText = v; else e.setAttribute(k, v); }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}
function sv(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
function svText(x, y, text, attrs, fill) {
  const t = sv("text", Object.assign({x, y}, attrs || {})); t.textContent = text; t.style.fill = fill || "var(--ink-2)"; return t;
}
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const mcol = m => css(M[m].cv);
const ok = x => x != null && isFinite(x);
const pct = x => ok(x) ? String(Math.round(x * 100)).replace("-", "−") + "%" : "—";
const pct1 = x => ok(x) ? (Math.abs(x) < 0.1 ? (x * 100).toFixed(1) : Math.round(x * 100)) + "%" : "—";
const sgn = (x, d = 3) => !ok(x) ? "—" : Math.abs(x) < 0.5 * Math.pow(10, -d) ? (0).toFixed(d) : (x > 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const pctS = x => Number((x * 100).toPrecision(2)).toString() + "%";
const chartW = box => Math.max(320, Math.floor(box.clientWidth || 520));
const iouT = x => ok(x) ? x.toFixed(3) : "—";
const dsName = ds => DSN[ds] || ds;
const isSrc = ds => ds === D.meta.source;

// ---------- data access ----------
function val(ds, site, m) {
  if (m === "oracle") return ((D.oracle[ds] || {})[site]) || null;
  return (((D.cal[ds] || {})[site] || {})[m]) || null;
}
function auc(ds, site) { const p = D.probe[site]; return p ? p.auc[ds] : null; }
function offBad(ds, site) { const p = D.probe[site]; return p ? p.off_bad[ds] : null; }
function methodsFor(site) {
  const s = SITE[site]; const out = ["dm"];
  if (s.type === "residual") out.push("mean");
  out.push("dm_rand", "dm_rev");
  if (s.kind === "logits") out.push("const");
  out.push("oracle");
  return out;
}

// ---------- colour ----------
const SEQ = {light: ["#e9f1fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"],
             dark:  ["#1f2a3a", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"]};
const scheme = () => css("--scheme") === "dark" ? "dark" : "light";
function hex2rgb(h) { h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
function rgb2hex(c) { return "#" + c.map(v => Math.round(v).toString(16).padStart(2, "0")).join(""); }
function ramp(t) {
  const stops = SEQ[scheme()]; t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[i + 1]);
  return rgb2hex(a.map((v, k) => v + (b[k] - v) * f));
}
function textOn(hex) {
  const [r, g, b] = hex2rgb(hex).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 ? "#0b0b0b" : "#ffffff";
}
function nodeValue(site) {
  if (st.c === "auc") { const a = auc(st.ds, site); return ok(a) ? {t: (a - 0.5) / 0.5, label: a.toFixed(2)} : null; }
  const v = val(st.ds, site, st.m); return v && ok(v.restore) ? {t: v.restore, label: pct(v.restore)} : null;
}

// ---------- tooltip ----------
const tip = $("tip");
function showTip(ev, rows) {
  tip.replaceChildren(...rows.map(([c, t]) => el("div", c ? {class: c} : null, t)));
  tip.hidden = false; moveTip(ev);
}
function moveTip(ev) {
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = innerHeight - r.height - 8;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };
function siteTip(id) {
  const n = NODE[id], s = SITE[id];
  const rows = [["", (s ? id + " · " : "") + n.label], ["l", n.code]];
  if (!s) { rows.push(["l", n.kind === "const" ? "не зависит от бокса" : "сюда вектор не добавлялся"]); return rows; }
  for (const m of methodsFor(id)) { const v = val(st.ds, id, m); rows.push(["v", M[m].name + ": " + (v ? pct(v.restore) : "—")]); }
  rows.push(["v", "AUC пробы: " + (ok(auc(st.ds, id)) ? auc(st.ds, id).toFixed(2) : "—")]);
  return rows;
}

// ---------- graph ----------
const SUCC = {};
for (const n of S.nodes) SUCC[n.id] = [];
for (const e of S.edges) SUCC[e.src].push(e.dst);
function reach(id) {
  const seen = new Set([id]), stack = [id];
  while (stack.length) for (const y of SUCC[stack.pop()]) if (!seen.has(y)) { seen.add(y); stack.push(y); }
  return seen;
}
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
  for (const e of edges) {
    if (pos[e.src].y === pos[e.dst].y) continue;
    (inc[e.dst] = inc[e.dst] || []).push(e); (out[e.src] = out[e.src] || []).push(e);
  }
  for (const es of Object.values(inc)) { es.sort((a, b) => pos[a.src].x - pos[b.src].x || pos[a.src].y - pos[b.src].y); es.forEach((e, i) => pin[e.id] = (i - (es.length - 1) / 2) * 14); }
  for (const es of Object.values(out)) { es.sort((a, b) => pos[a.dst].x - pos[b.dst].x || pos[a.dst].y - pos[b.dst].y); es.forEach((e, i) => pout[e.id] = (i - (es.length - 1) / 2) * 14); }
  return {pin, pout};
}
function edgePath(e, pos, pp) {
  const a = pos[e.src], b = pos[e.dst];
  if (a.y === b.y) {
    const right = a.x < b.x, sx = right ? a.x + a.w : a.x, ex = right ? b.x - 3 : b.x + b.w + 3, y = a.y + a.h / 2;
    return `M${sx},${y} L${ex},${y}`;
  }
  const sx = a.x + a.w / 2 + (pp.pout[e.id] || 0), sy = a.y + a.h, ex = b.x + b.w / 2 + (pp.pin[e.id] || 0), ey = b.y - 3;
  const k = Math.min(30, (ey - sy) / 2);
  return `M${sx},${sy} C${sx},${sy + k} ${ex},${ey - k} ${ex},${ey}`;
}
function drawGraph(scroll) {
  const L = layout();
  const svg = sv("svg", {width: L.W, height: L.H, viewBox: `0 0 ${L.W} ${L.H}`, role: "img",
    "aria-label": "Граф SAM: prompt encoder и mask decoder; узлы, куда можно добавить вектор, окрашены"});
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
    eg.appendChild(p);
  }
  svg.appendChild(eg);
  let selY = 0;
  for (const n of S.nodes) {
    const p = L.pos[n.id], site = SITE[n.id], sel = n.id === st.site;
    const g = sv("g", {class: "node" + (site ? " site" : ""), transform: `translate(${p.x},${p.y})`});
    if (site) { g.setAttribute("tabindex", 0); g.setAttribute("role", "button"); g.setAttribute("aria-pressed", sel); g.setAttribute("aria-label", n.id + ": " + n.label); }
    if (!site && !down.has(n.id)) g.style.opacity = 0.5;
    const r = sv("rect", {class: "frame", width: p.w, height: p.h, rx: 7});
    let ink = "var(--ink)", nv = null;
    if (site) {
      nv = nodeValue(n.id);
      const fill = nv ? ramp(nv.t) : "var(--surface-2)";
      r.style.fill = fill; if (nv) ink = textOn(fill);
      r.style.stroke = sel ? "var(--sel)" : "var(--ink-2)"; r.style.strokeWidth = sel ? 3 : 1.2;
    } else {
      r.style.fill = n.kind === "const" ? "url(#hatch)" : "var(--surface)";
      r.style.stroke = "var(--grid)"; r.style.strokeWidth = 1;
    }
    g.appendChild(r);
    g.appendChild(svText(8, 16, n.label, {"font-size": 11.5, "font-weight": site ? 650 : 500}, site ? ink : "var(--ink-2)"));
    if (site) {
      g.appendChild(svText(8, 32, n.id, {"font-size": 10.5}, ink));
      g.appendChild(svText(p.w - 8, 32, nv ? nv.label : "—", {"font-size": 12, "font-weight": 650, "text-anchor": "end"}, ink));
    }
    if (sel) {
      selY = p.y;
      const b = sv("g", {transform: `translate(${-6},${p.h / 2})`});
      const c = sv("rect", {x: -36, y: -10, width: 36, height: 20, rx: 10}); c.style.fill = "var(--sel)"; b.appendChild(c);
      b.appendChild(svText(-18, 4, "+αv", {"font-size": 11, "font-weight": 700, "text-anchor": "middle"}, "var(--bg)"));
      g.appendChild(b);
    }
    g.addEventListener("pointerenter", ev => showTip(ev, siteTip(n.id)));
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
  const stops = Array.from({length: 11}, (_, i) => ramp(i / 10) + " " + i * 10 + "%").join(",");
  const r = el("span", {class: "ramp", style: "background:linear-gradient(90deg," + stops + ")"});
  if (st.c === "auc") lg.append(el("span", null, "AUC пробы bad vs контроль: 0.5", r, "1.0"));
  else lg.append(el("span", null, "восстановление (" + M[st.m].name + "): 0%", r, "100%"));
  lg.append(el("span", null, "клик по цветному узлу — добавить вектор туда; тёмные рёбра — куда вектор дальше распространяется, бледные узлы и рёбра он не затрагивает"));
}

// ---------- charts ----------
function axisY(svg, x0, x1, y, ticks, fmt) {
  for (const t of ticks) {
    const yy = y(t);
    const l = sv("line", {x1: x0, x2: x1, y1: yy, y2: yy}); l.style.stroke = t === 0 ? "var(--axis)" : "var(--grid)"; l.style.strokeWidth = 1; svg.appendChild(l);
    svg.appendChild(svText(x0 - 6, yy + 4, fmt(t), {"font-size": 11, "text-anchor": "end"}, "var(--muted)"));
  }
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
function key(items) {
  return el("div", {class: "ckey"}, ...items.map(([label, color, dash]) =>
    el("span", null, el("i", {style: `border-top:2px ${dash ? "dashed" : "solid"} ${color}`}), label)));
}
function drawLayers() {
  const box = $("ch-layers"); box.replaceChildren();
  const ids = D.sites.map(s => s.id), ds = st.ds;
  const lines = ["dm", "mean", "dm_rand", "oracle"];
  box.append(el("h3", null, "Восстановление по узлам · " + dsName(ds) + (isSrc(ds) ? ", вне фолда" : ", перенос")));
  box.append(key(lines.map(m => [M[m].name, mcol(m), m === "oracle"]).concat([["AUC пробы (нижняя панель)", css("--ink-2"), false]])));
  const W = chartW(box), ml = 42, mr = 10, mt = 8, h1 = 180, gap = 30, h2 = 66, mb = 62;
  const svg = sv("svg", {width: W, height: mt + h1 + gap + h2 + mb, viewBox: `0 0 ${W} ${mt + h1 + gap + h2 + mb}`, role: "img", "aria-label": "Восстановление по узлам и AUC пробы"});
  const x = i => ml + (i + 0.5) * (W - ml - mr) / ids.length;
  const vals = lines.flatMap(m => ids.map(s => (val(ds, s, m) || {}).restore)).filter(ok);
  const lo = Math.min(0, ...vals), hi = Math.max(1, ...vals);
  const y = v => mt + h1 - (v - lo) / (hi - lo) * h1;
  const si = ids.indexOf(st.site), bw = (W - ml - mr) / ids.length;
  const band = sv("rect", {x: x(si) - bw / 2, y: mt, width: bw, height: h1 + gap + h2}); band.style.fill = "var(--surface-2)"; svg.appendChild(band);
  axisY(svg, ml, W - mr, y, [0, 0.25, 0.5, 0.75, 1].filter(t => t >= lo && t <= hi), t => Math.round(t * 100) + "%");
  for (const m of lines) {
    const pts = ids.map((s, i) => { const v = val(ds, s, m); return [x(i), v && ok(v.restore) ? y(v.restore) : null]; });
    poly(svg, pts, mcol(m), m === "oracle" ? "5 4" : null, m === st.m ? 2.6 : 1.8);
  }
  for (const m of lines) ids.forEach((s, i) => {
    const v = val(ds, s, m); if (!v || !ok(v.restore)) return;
    dot(svg, x(i), y(v.restore), mcol(m), [["", s + " · " + M[m].name], ["v", "восстановление " + pct(v.restore) + (ok(v.lo) ? "  [" + pct(v.lo) + " – " + pct(v.hi) + "]" : "")], ["l", v.cfg || ""]], () => select(s), m === st.m && s === st.site);
  });
  const y2 = v => mt + h1 + gap + h2 - (v - 0.5) / 0.5 * h2;
  axisY(svg, ml, W - mr, y2, [0.5, 0.75, 1], t => t.toFixed(2));
  svg.appendChild(svText(ml, mt + h1 + gap - 8, "AUC пробы: bad vs контроль", {"font-size": 11}, "var(--ink-2)"));
  const ap = ids.map((s, i) => [x(i), ok(auc(ds, s)) ? y2(Math.max(0.5, auc(ds, s))) : null]);
  poly(svg, ap, "var(--ink-2)", null, 1.6);
  ids.forEach((s, i) => { const a = auc(ds, s); if (ok(a)) dot(svg, x(i), y2(Math.max(0.5, a)), "var(--ink-2)", [["", s], ["v", "AUC " + a.toFixed(3)], ["l", ok(offBad(ds, s)) ? "bad вне области хороших: " + pct(offBad(ds, s)) : ""]], () => select(s)); });
  ids.forEach((s, i) => {
    const t = svText(x(i) + 3, mt + h1 + gap + h2 + 14, s, {"font-size": 11, "text-anchor": "end", transform: `rotate(-40 ${x(i) + 3} ${mt + h1 + gap + h2 + 14})`}, s === st.site ? "var(--ink)" : "var(--muted)");
    t.style.cursor = "pointer"; if (s === st.site) t.setAttribute("font-weight", 700);
    t.addEventListener("click", () => select(s)); svg.appendChild(t);
  });
  box.append(svg);
  box.append(el("p", {class: "note"}, "Бюджет: хорошие маски меняются не больше чем на " + pct1(D.meta.budget) + ". Oracle — потолок для этого узла, подмена тензора из best той же пары."));
}
function drawDose() {
  const box = $("ch-dose"); box.replaceChildren();
  const site = st.site, d = D.dose[site] || {};
  box.append(el("h3", null, "Доза → эффект в " + site + " · " + dsName(D.meta.source) + ", вне фолда"));
  const ms = ["dm", "mean", "dm_rand", "dm_rev", "const"].filter(m => d[m]);
  if (!ms.length) { box.append(el("p", {class: "note"}, "нет данных")); return; }
  box.append(key(ms.map(m => [M[m].name + (d[m].group ? " · " + d[m].group : ""), mcol(m), false])));
  const W = chartW(box), H = 300, ml = 42, mr = 12, mt = 8, mb = 40;
  const svg = sv("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Восстановление против изменения хороших масок по α"});
  const pts = ms.flatMap(m => d[m].pts);
  const xs = pts.map(p => Math.max(1e-4, p[1] || 1e-4));
  const lx0 = Math.floor(Math.log10(Math.min(...xs))), lx1 = Math.max(lx0 + 1, Math.ceil(Math.log10(Math.max(...xs))));
  const ys = pts.map(p => p[2]).filter(ok);
  const lo = Math.min(0, ...ys), hi = Math.max(0.5, ...ys);
  const x = v => ml + (Math.log10(Math.max(1e-4, v)) - lx0) / (lx1 - lx0) * (W - ml - mr);
  const y = v => mt + (H - mt - mb) - (v - lo) / (hi - lo) * (H - mt - mb);
  axisY(svg, ml, W - mr, y, [-1, -0.5, 0, 0.25, 0.5, 0.75, 1].filter(t => t >= lo && t <= hi), t => Math.round(t * 100) + "%");
  for (let e = lx0; e <= lx1; e++) for (const f of [1, 2, 5]) {
    const v = f * Math.pow(10, e); if (v > Math.pow(10, lx1) + 1e-12) continue;
    const xx = x(v), l = sv("line", {x1: xx, x2: xx, y1: mt, y2: H - mb}); l.style.stroke = "var(--grid)"; l.style.strokeWidth = f === 1 ? 1 : 0.6; svg.appendChild(l);
    if (f === 1) svg.appendChild(svText(xx, H - mb + 14, pctS(v), {"font-size": 10.5, "text-anchor": "middle"}, "var(--muted)"));
  }
  const bx = x(D.meta.budget), bl = sv("line", {x1: bx, x2: bx, y1: mt, y2: H - mb}); bl.style.stroke = "var(--ink-2)"; bl.setAttribute("stroke-dasharray", "3 3"); svg.appendChild(bl);
  svg.appendChild(svText(bx + 4, mt + 10, "бюджет", {"font-size": 10.5}, "var(--ink-2)"));
  svg.appendChild(svText(ml + (W - ml - mr) / 2, H - 6, "изменение хороших масок (лог. шкала)", {"font-size": 11, "text-anchor": "middle"}, "var(--ink-2)"));
  for (const m of ms) poly(svg, d[m].pts.map(p => [x(p[1] || 1e-4), ok(p[2]) ? y(p[2]) : null]), mcol(m), null, m === st.m ? 2.6 : 1.8);
  for (const m of ms) for (const p of d[m].pts) if (ok(p[2]))
    dot(svg, x(p[1] || 1e-4), y(p[2]), mcol(m), [["", M[m].name + " · α = " + p[0]], ["v", "восстановление " + pct(p[2]) + (ok(p[3]) ? "  [" + pct(p[3]) + " – " + pct(p[4]) + "]" : "")], ["v", "хорошие маски изменились на " + pct1(p[1])]]);
  box.append(svg);
  box.append(el("p", {class: "note"}, "Каждая точка — одна сила α для одной группы токенов. Хорошее направление лежит выше и левее: больше восстановления при меньшем вреде."));
}

// ---------- site panel ----------
function drawSite() {
  const s = SITE[st.site], ds = st.ds, box = $("site"); box.replaceChildren();
  box.append(el("h2", null, s.id + " · " + s.title));
  box.append(el("div", null, el("span", {class: "tag"}, s.type === "action" ? "тип 2 · action-head steering" : "тип 1 · residual steering"),
                el("span", {class: "mono muted"}, s.module)));
  box.append(el("p", {class: "lead"}, s.text));
  const v = val(ds, s.id, st.m), rnd = val(ds, s.id, "dm_rand"), orc = val(ds, s.id, "oracle");
  const where = dsName(ds) + (isSrc(ds) ? " (вне фолда)" : " (векторы и α заморожены по " + dsName(D.meta.source) + ")");
  let lead;
  if (!methodsFor(s.id).includes(st.m)) lead = M[st.m].name + " для этого узла не определён: Mean есть только у residual-узлов.";
  else if (!v || !ok(v.restore)) lead = M[st.m].name + ": ни одна сила α не укладывается в бюджет — хорошие маски меняются больше чем на " + pct1(D.meta.budget) + ".";
  else if (st.m === "oracle") lead = "На " + where + " подмена тензора из best той же пары возвращает " + pct(v.restore) + " потерянного IoU. Это потолок для любого вектора в этом узле.";
  else lead = "На " + where + " " + M[st.m].name + " возвращает " + pct(v.restore) + " [" + pct(v.lo) + " – " + pct(v.hi) + "] потерянного IoU" +
    (st.m !== "dm_rand" && rnd && ok(rnd.restore) ? "; случайное направление той же нормы — " + pct(rnd.restore) : "") +
    (orc && ok(orc.restore) ? "; потолок (oracle) — " + pct(orc.restore) : "") + ". Хорошие маски меняются на " + pct1(v.chg_good) + " пикселей.";
  box.append(el("p", null, lead));
  const tb = el("table");
  tb.append(el("thead", null, el("tr", null, el("th", null, "Метод"), el("th", {class: "num"}, "Восстановление"),
    el("th", {class: "num"}, "Починено ≥ 50%"), el("th", {class: "num"}, "Изм. хороших масок"),
    el("th", {class: "num"}, "ΔIoU боксов у GT"), el("th", null, "Конфигурация"))));
  const body = el("tbody");
  for (const m of methodsFor(s.id)) {
    const x = val(ds, s.id, m), pick = PICK.includes(m);
    const tr = el("tr", {class: (m === st.m ? "cur " : "") + (pick ? "pick" : "")},
      el("td", null, el("span", {class: "dot", style: "background:" + mcol(m) + ";margin-right:6px"}), M[m].name, el("div", {class: "muted", style: "font-size:11px"}, M[m].long)),
      el("td", {class: "num"}, x && ok(x.restore) ? pct(x.restore) : "—", x && ok(x.lo) ? el("div", {class: "muted", style: "font-size:11px"}, pct(x.lo) + " – " + pct(x.hi)) : null),
      el("td", {class: "num"}, x ? pct(x.fixed) : "—"),
      el("td", {class: "num"}, x ? pct1(x.chg_good) : "—"),
      el("td", {class: "num"}, x ? sgn(x.d_dense) : "—"),
      el("td", {style: "font-size:12px"}, x ? x.cfg : ""));
    if (pick) { tr.tabIndex = 0; tr.addEventListener("click", () => setM(m)); tr.addEventListener("keydown", ev => { if (ev.key === "Enter") setM(m); }); }
    body.append(tr);
  }
  tb.append(body);
  box.append(el("div", {class: "tbl"}, tb));
  const a = auc(ds, s.id), ob = offBad(ds, s.id);
  box.append(el("p", {class: "note"}, "AUC пробы bad vs контроль: " + (ok(a) ? a.toFixed(2) : "—") +
    " · bad-состояний вне области хороших: " + pct(ob) + " · бюджет вреда: " + pct1(D.meta.budget) + " изменённых пикселей хороших масок."));
}

// ---------- examples ----------
const imgCache = new Map(), layerCache = new Map();
function loadImg(b64, mime) {
  const k = mime + b64.length + b64.slice(0, 40) + b64.slice(-40);
  if (!imgCache.has(k)) { const im = new Image(); im.src = "data:" + mime + ";base64," + b64; imgCache.set(k, im.decode().then(() => im)); }
  return imgCache.get(k);
}
const MASK_RGB = [227, 52, 52];
async function maskLayer(b64, w, h) {
  const k = b64;
  if (layerCache.has(k)) return layerCache.get(k);
  const im = await loadImg(b64, "image/png");
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const x = c.getContext("2d"); x.drawImage(im, 0, 0, w, h);
  const d = x.getImageData(0, 0, w, h), p = d.data, on = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) on[i] = p[i * 4] > 127 ? 1 : 0;
  for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) {
    const i = yy * w + xx, j = i * 4;
    if (!on[i]) { p[j + 3] = 0; continue; }
    const edge = xx === 0 || yy === 0 || xx === w - 1 || yy === h - 1 || !on[i - 1] || !on[i + 1] || !on[i - w] || !on[i + w];
    p[j] = MASK_RGB[0]; p[j + 1] = MASK_RGB[1]; p[j + 2] = MASK_RGB[2]; p[j + 3] = edge ? 255 : 105;
  }
  x.putImageData(d, 0, 0);
  layerCache.set(k, c);
  return c;
}
async function paint(cv, ex, maskB64, boxPx, token) {
  const base = await loadImg(ex.base, "image/jpeg");
  const layer = maskB64 ? await maskLayer(maskB64, ex.w, ex.h) : null;
  if (token !== renderToken) return;
  cv.width = ex.w; cv.height = ex.h;
  const x = cv.getContext("2d");
  x.drawImage(base, 0, 0, ex.w, ex.h);
  if (layer) x.drawImage(layer, 0, 0);
  const [x0, y0, x1, y1] = boxPx;
  x.lineWidth = 3.5; x.strokeStyle = "rgba(0,0,0,0.65)"; x.strokeRect(x0, y0, x1 - x0, y1 - y0);
  x.lineWidth = 1.6; x.strokeStyle = "#ffffff"; x.strokeRect(x0, y0, x1 - x0, y1 - y0);
}
let renderToken = 0;
function panel(ex, title, cap, maskB64, boxPx, token) {
  const cv = el("canvas", {width: ex.w, height: ex.h, role: "img", "aria-label": title});
  paint(cv, ex, maskB64, boxPx, token);
  return el("div", {class: "pn"}, cv, el("div", {class: "c"}, title, el("br"), cap));
}
function naPanel(ex, title, text) {
  return el("div", {class: "pn"}, el("div", {class: "na", style: `aspect-ratio:${ex.w}/${ex.h}`}, text), el("div", {class: "c"}, title));
}
function iouCap(iou, base) {
  const d = ok(iou) && ok(base) ? iou - base : null;
  return el("span", null, "IoU ", el("b", null, iouT(iou)),
    d != null && Math.abs(d) >= 0.0005 ? el("span", {class: d > 0 ? "up" : "down"}, "  " + sgn(d)) : null);
}
function drawExamples() {
  const box = $("examples"); box.replaceChildren();
  const list = (D.examples || {})[st.ds] || [];
  const token = ++renderToken;
  const head = el("div", {class: "ex-head"}, el("h2", {style: "margin:0"}, "Маски до и после: " + dsName(st.ds) + " · " + st.site + " · " + M[st.m].name));
  head.append(el("span", {class: "note", style: "margin:0"},
    el("span", {class: "swatch", style: "background:rgb(227,52,52)"}), "маска SAM   ",
    el("span", {class: "swatch", style: "background:rgb(60,220,60)"}), "контур GT   ",
    el("span", {class: "swatch", style: "background:#fff;border:1px solid #000"}), "бокс"));
  box.append(head);
  if (!D.meta.examples_computed) { box.append(el("p", {class: "note"}, "Примеры не посчитаны: отчёт собран с --no_examples.")); return; }
  if (!list.length) { box.append(el("p", {class: "note"}, "Для этого набора нет примеров.")); return; }
  box.append(el("p", {class: "note", style: "margin:0 0 8px"}, "Пары взяты по квантилям восстановления лучшего узла (" + D.headline.site + ", DM), чтобы были видны и удачи, и провалы. Маски пересчитаны с теми же боксами, векторами и α, что в прогоне, и сверены с его таблицами."));
  const grid = el("div", {class: "exs"});
  for (const ex of list) {
    const s = ((ex.steer || {})[st.site] || {})[st.m];
    const t = el("div", {class: "t"}, el("b", null, ex.image), "  · пара " + ex.pid + " · сдвиг бокса " + ex.shift + " px" + (ex.fold >= 0 ? " · фолд " + ex.fold : ""));
    if (s) t.append("  · ", el("b", null, M[st.m].name + ": " + s.cfg + (ok(s.restore) ? " → восстановлено " + pct(s.restore) : "")));
    const ps = el("div", {class: "panels"});
    ps.append(panel(ex, "best бокс", iouCap(ex.iou_best0), ex.m_best0, ex.box_best, token));
    ps.append(panel(ex, "bad бокс", iouCap(ex.iou_bad0), ex.m_bad0, ex.box_bad, token));
    if (!methodsFor(st.site).includes(st.m)) {
      ps.append(naPanel(ex, "bad + вектор", "метод не определён для этого узла"));
      ps.append(naPanel(ex, "best + вектор", ""));
    } else if (!s) {
      ps.append(naPanel(ex, "bad + вектор", "нет α в бюджете вреда"));
      ps.append(naPanel(ex, "best + вектор", ""));
    } else {
      ps.append(panel(ex, "bad + " + (st.m === "oracle" ? "тензор из best" : "вектор"), iouCap(s.iou_bad, ex.iou_bad0), s.m_bad, ex.box_bad, token));
      ps.append(panel(ex, "best + " + (st.m === "oracle" ? "подмена (не трогает best)" : "вектор"),
        s.m_best ? iouCap(s.iou_best, ex.iou_best0) : el("span", null, "маска не изменилась"), s.m_best || ex.m_best0, ex.box_best, token));
    }
    grid.append(el("div", {class: "ex"}, t, ps));
  }
  box.append(grid);
}

// ---------- header, controls, checks ----------
function drawHeader() {
  const m = D.meta;
  $("meta").textContent = m.model + " · " + m.checkpoint + " · источник " + dsName(m.source) + " (" + (m.images[m.source] || "?") + " картинок, " +
    (m.pairs[m.source] || "?") + " пар, " + m.folds + " фолдов)" + D.datasets.filter(d => !isSrc(d.name)).map(d => " · перенос " + dsName(d.name) + " (" + (m.pairs[d.name] || "?") + " пар)").join("") +
    " · прогон " + m.created + " · отчёт " + m.report_created;
  const h = D.headline, src = m.source, chips = $("chips"); chips.replaceChildren();
  const dm = val(src, h.site, "dm"), rn = val(src, h.site, "dm_rand");
  if (dm) chips.append(el("div", {class: "chip"}, "Лучший узел для DM на " + dsName(src) + ": ", el("b", null, h.site), " — восстановлено ", el("b", null, pct(dm.restore)),
    rn && ok(rn.restore) ? " против " + pct(rn.restore) + " у случайного направления той же нормы" : ""));
  if (h.first_auc) chips.append(el("div", {class: "chip"}, "Проба впервые отличает bad от контролей (AUC ≥ 0.65) в ", el("b", null, h.first_auc), ": " + auc(src, h.first_auc).toFixed(2)));
  const tr = D.datasets.filter(d => !isSrc(d.name)).map(d => { const v = val(d.name, h.site, "dm"); return v && ok(v.restore) ? dsName(d.name) + " " + pct(v.restore) : null; }).filter(Boolean);
  if (tr.length) chips.append(el("div", {class: "chip"}, "Перенос " + h.site + " без подстройки: ", el("b", null, tr.join(", "))));
  chips.append(el("div", {class: "chip"}, "Бюджет вреда: хорошие маски меняются ≤ ", el("b", null, pct1(m.budget))));
}
function seg(id, items, cur, onpick) {
  const s = $(id); s.replaceChildren();
  for (const [v, label, color, disabled] of items) {
    const b = el("button", {type: "button", "aria-pressed": String(v === cur)});
    if (color) b.append(el("span", {class: "dot", style: "background:" + color}));
    b.append(label); if (disabled) b.disabled = true;
    b.addEventListener("click", () => onpick(v)); s.append(b);
  }
}
function drawControls() {
  seg("seg-ds", D.datasets.map(d => [d.name, dsName(d.name) + (d.role === "source" ? " · источник" : " · перенос")]), st.ds, v => { st.ds = v; render(); });
  seg("seg-m", PICK.map(m => [m, M[m].name, mcol(m), !methodsFor(st.site).includes(m)]), st.m, setM);
  seg("seg-c", [["restore", "восстановление"], ["auc", "AUC пробы"]], st.c, v => { st.c = v; render(); });
}
function drawChecks() {
  const c = D.run_checks, r = D.report_checks, b = $("checks-body"); b.replaceChildren();
  const li = (cls, text) => el("li", {class: cls}, text);
  const ul = el("ul");
  ul.append(li("ok", "нулевая доза воспроизводит прогон без вектора бит в бит: " + c.zero_dose));
  ul.append(li("ok", "steering через hook = trace_forward с подменённым тензором: " + c.trace));
  ul.append(li(c.noop === 0 ? "ok" : "warn", "вектор на неиспользуемых токенах ничего не меняет: max rel " + c.noop));
  ul.append(li(c.baseline === 0 ? "ok" : "warn", "повтор прогона без вектора: max |ΔIoU| " + c.baseline));
  ul.append(li("", "oracle на разрезах графа (должно быть 1.0): " + Object.entries(c.oracle_cuts).map(([k, v]) => k + " " + v).join(", ")));
  ul.append(li("", "воспроизведены пары: " + Object.entries(c.repro).map(([k, v]) => dsName(k) + " " + Object.entries(v).map(([s, n]) => n + " " + s).join(" / ")).join(", ")));
  if (r) {
    const good = r.iou_max_dev <= D.meta.iou_tol && r.missing_rows === 0;
    ul.append(li(good ? "ok" : "fail", "маски примеров пересчитаны и сверены с таблицами прогона: " + r.iou_checked + " значений IoU, max |разница| " + r.iou_max_dev.toExponential(1) + (r.iou_max_where ? " (" + r.iou_max_where + ")" : "") + (r.missing_rows ? ", без строки в таблице: " + r.missing_rows : "")));
    ul.append(li(r.sha_fold_ok && r.sha_all_ok ? "ok" : "warn", "векторы совпадают с прогоном (sha256): по фолдам " + r.sha_fold_ok + ", общий " + r.sha_all_ok + "; фолды картинок совпадают: " + r.fold_assign_ok));
  }
  b.append(ul);
}

// ---------- state ----------
function setM(m) { if (!methodsFor(st.site).includes(m)) return; st.m = m; render(); }
function select(site) {
  st.site = site;
  if (!methodsFor(site).includes(st.m)) st.m = "dm";
  render(true);
}
function writeHash() { history.replaceState(null, "", "#" + new URLSearchParams({ds: st.ds, m: st.m, site: st.site, c: st.c})); }
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (D.datasets.some(d => d.name === p.get("ds"))) st.ds = p.get("ds");
  if (SITE[p.get("site")]) st.site = p.get("site");
  if (PICK.includes(p.get("m")) && methodsFor(st.site).includes(p.get("m"))) st.m = p.get("m");
  if (["restore", "auc"].includes(p.get("c"))) st.c = p.get("c");
}
function render(scroll) { hideTip(); drawControls(); drawLegend(); drawGraph(scroll); drawSite(); drawLayers(); drawDose(); drawExamples(); writeHash(); }
$("btn-terms").addEventListener("click", () => { const t = $("terms"); t.open = true; t.scrollIntoView({behavior: "smooth"}); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render());
let rsz = 0, lastW = 0;
addEventListener("resize", () => { clearTimeout(rsz); rsz = setTimeout(() => { const w = $("ch-dose").clientWidth; if (w !== lastW) { lastW = w; drawLayers(); drawDose(); } }, 150); });
readHash(); drawHeader(); drawChecks(); render(true);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
