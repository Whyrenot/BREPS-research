"""
steer_universal.py
==================
One steering vector for every dataset? Every user-study subset gives its own
vector on one half of its images; their mean is tried as the defence on the
other halves -- for SAM-1, SAM-2.1 and SAM-3 with the same code.

The scans (steer_models_scan.py) and the pool (build_candidate_pool.py) steer
with the COCO A vector only. steer_variants.py found no universal vector for
SAM-1 at L2.norm2 from 10-image vectors per dataset; here every subset gives
the 25 images of its half A, the site is the pool's (L2.norm3), the alpha
policy is also fitted on all subsets, and all three models run.

PROTOCOL
  data     the user study FOR_TEST, --use m, all ten subsets (50 images x 25
           annotations); box = the tight bbox of the user's mask, as everywhere
  split    every subset in halves A / B by steer_user_study.split_halves
           (--split_seed; COCO's are the scan's, checked): A estimates the
           vectors and fits the alpha policies, B is every subset's held-out test
  vector   per subset, at --site, all tokens: the mean over its A images of
           mean(top50 states) - mean(bottom50 states) inside the image (the
           scan's rule; groups by multimask=False IoU, the round(n / 2) worst)
  variants (every alpha of --alphas, both output modes from one decoder pass)
    own        the subset's own vector
    coco       the COCO A vector = the scan's and the pool's (checked)
    pool       the mean of all subsets' vectors, each subset weighted equally:
               the universal vector
    pool_norm  the same mean of the vectors scaled to unit norm first (no
               subset dominates by its norm)
    pool_nat   the mean of the natural subsets' vectors (medicine left out)
    rand       a fixed random direction with pool's norm per token: the control
    lodo       leave one dataset out: the mean of the OTHER subsets' vectors --
               the vector for a domain never seen
  norm     every variant is rescaled to the norm of the COCO A vector (--norm
           coco), so one alpha is one dose for all of them and the pool's grid
           fits all; --norm raw keeps their own norms. The factor is the full
           vector's; its out-of-fold versions get the same one
  cross-fitting  an A image is steered with variants in which its own subset's
           vector is the one of the other folds of that subset's A (the scan's
           folds and seed); the other subsets' vectors are full (they do not
           contain the image), so every alpha policy learns from out-of-sample
           effects. On A, lodo runs for each of the other subsets (the training
           rows of their lodo vectors), on B for the image's own subset
  With --use other than the scan's, COCO's vector is COCO A on those
  annotations and differs from the scan's (the check is then reported only).

ALPHA POLICIES (--from_dump: CPU, the main env; per output mode)
  alpha*       one alpha (or none), the best mean gain on the training rows
  GBM          steer_gate's gain regressor ("out" features: predicted IoU,
               stability, box fill, what each alpha does to the mask and to the
               predicted IoU) -- the paper's alpha regressor
  alpha* own   one alpha from the test subset's own A rows: a universal
               direction with the domain's dose (pool*, coco)
  oracle       the best single alpha on the test half itself (hindsight): the
               direction at its best dose
  training rows: own -- the subset's A; coco -- COCO A only (the current
  practice); coco_allA, pool, pool_norm, rand -- the A halves of all subsets;
  pool_nat -- the natural A halves; lodo -- the A halves of the other subsets

TABLES (the B halves; % to undef; means over subsets, each weighted equally,
95% image-bootstrap CIs by steer_gate.score_blocks): every method per subset
and natural / medical / all; pairs with paired CIs (pool vs own, pool vs coco,
lodo vs coco, pool vs rand, ...); the dose response; alpha* per variant; the
vectors' geometry (cosines between subsets, the split-half noise level, cos
with the pool and with the leave-one-out vector).

CHECKS (checks.json; * stop the run)
  * the scan is this model's run (decoder layout, COCO split A)
  * COCO's own vector and its fold vectors = the scan's vectors.npz (same --use)
  * zero dose bit for bit on every image; own decoder pass = MaskDecoder.forward
    on the first A image of every subset (public API reported)
  * halves disjoint, n // 2 images in A
  groups round(n / 2) per image; pass 2's A baseline = pass 1's; undef /
  head-select vs --reference_csv (reported). TF32 matmuls off (sam3 turns them on).

OUTPUTS (--out_dir)
  rows.csv.gz       one row per box and variant: steer_multimask's dump schema
                    (tok0_* = mm0, multi_* = mm1) + variant, group, site, table
                    (<subset>_A / _B), role (train = A)
  vectors.npz       own/<subset>, own/<subset>/fold<k> (raw), var/<vector> (the
                    full vectors as steered), dimg/<subset> (per-image DMs),
                    stems/<subset>
  vector_stats.csv, split.json, checks.json, meta.json
  --from_dump adds  table.csv, pairs.csv, alpha_curve.csv, alpha_star.csv,
                    geometry.csv, cos_own.csv, summary.md

    CUDA_VISIBLE_DEVICES=0 python scripts/steer_universal.py --model_name SAM \\
        --root /.../FOR_TEST --checkpoint_path /.../sam_vit_b_01ec64.pth \\
        --scan_dir exp_res/steer_scan_SAM --reference_csv results/samfull_m.csv \\
        --out_dir exp_res/steer_universal/SAM_L2.norm3
    python scripts/steer_universal.py --from_dump exp_res/steer_universal/SAM_L2.norm3

    # all three models, one per GPU, then the tables
    GPUS="0 1 2" bash scripts/run_steer_universal.sh
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heatmaps.env_dispatch import maybe_dispatch_to_env  # noqa: E402

EPS = 1e-12
MODES = ((0, "tok0"), (1, "multi"))
VARIANTS = ("own", "coco", "pool", "pool_norm", "pool_nat", "rand", "lodo")
GROUPS = ("all", "bottom50", "top50")


def _a(alpha: float) -> str:
    return f"a{alpha:g}"


def _floats(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def cos(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), EPS))


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def combine(name: str, ds: dict, natural, u: np.ndarray) -> np.ndarray:
    """A variant's raw vector from the subsets' vectors ds {subset: (T, 256)}."""
    names = list(ds)
    if name == "coco":
        return ds["COCO"]
    if name == "pool":
        return np.mean([ds[d] for d in names], 0)
    if name == "pool_norm":
        return np.mean([ds[d] / max(np.linalg.norm(ds[d]), EPS) for d in names], 0)
    if name == "pool_nat":
        return np.mean([ds[d] for d in names if d in natural], 0)
    if name == "rand":
        return u * np.linalg.norm(combine("pool", ds, natural, u), axis=-1, keepdims=True)
    if name.startswith("lodo:"):
        x = name.split(":", 1)[1]
        return np.mean([ds[d] for d in names if d != x], 0)
    raise ValueError(name)


class Vectors:
    """Every subset's vector (all of its A and per fold) and the variants built
    from them, rescaled (--norm coco) by the factor of the full vector."""

    def __init__(self, dimg: dict, folds_of: dict, k: int, variants, natural, u: np.ndarray, norm: str,
                 allow_empty_fold: bool):
        self.datasets, self.variants, self.natural, self.u = list(dimg), list(variants), list(natural), u
        self.own = {d: x.mean(0).astype(np.float32) for d, (_, x) in dimg.items()}
        self.fold, self.empty_folds = {}, []
        for d, (stems, x) in dimg.items():
            f = np.array([folds_of[d][s] for s in stems])
            for j in range(k):
                sel = f != j
                if not sel.any():
                    if not allow_empty_fold:
                        raise RuntimeError(f"{d}: no A image outside fold {j}")
                    self.empty_folds.append(f"{d}/fold{j}")
                    sel = np.ones(len(f), bool)
                self.fold[(d, j)] = x[sel].mean(0).astype(np.float32)
        self.target = float(np.linalg.norm(self.own["COCO"]))
        self.scale = {}
        for key in self.keys():
            n = float(np.linalg.norm(self.raw(key, self.own)))
            self.scale[key] = self.target / max(n, EPS) if norm == "coco" else 1.0

    def keys(self) -> list:
        out = []
        for v in self.variants:
            if v == "own":
                out += [f"own:{d}" for d in self.datasets]
            elif v == "lodo":
                out += [f"lodo:{d}" for d in self.datasets]
            else:
                out.append(v)
        return out

    def raw(self, key: str, ds: dict) -> np.ndarray:
        if key.startswith("own:"):
            return ds[key.split(":", 1)[1]]
        return combine(key, ds, self.natural, self.u)

    def full(self) -> "OrderedDict[str, np.ndarray]":
        """Every vector as it is steered on the B halves."""
        return OrderedDict((key, (self.raw(key, self.own) * self.scale[key]).astype(np.float32)) for key in self.keys())

    def for_image(self, d: str, half: str, fold: int) -> "OrderedDict[str, np.ndarray]":
        """The variants an image is steered with -> {row label: vector}. A: its
        own subset's vector out of fold, lodo of every other subset; B: full,
        lodo of its own subset."""
        ds = dict(self.own)
        if half == "A":
            ds[d] = self.fold[(d, fold)]
        out = OrderedDict()
        for v in self.variants:
            if v == "own":
                out["own"] = ds[d] * self.scale[f"own:{d}"]
            elif v == "lodo":
                for x in ([d] if half == "B" else [e for e in self.datasets if e != d]):
                    out[f"lodo:{x}"] = self.raw(f"lodo:{x}", ds) * self.scale[f"lodo:{x}"]
            else:
                out[v] = self.raw(v, ds) * self.scale[v]
        return OrderedDict((n, v.astype(np.float32)) for n, v in out.items())


def vector_stats(vec: Vectors, h_norm: float) -> pd.DataFrame:
    full = vec.full()
    pool = combine("pool", vec.own, vec.natural, vec.u)
    rows = []
    for key, v in full.items():
        sub = key.split(":", 1)[1] if ":" in key else ""
        rows.append({"vector": key, "raw_norm": float(np.linalg.norm(vec.raw(key, vec.own))),
                     "scale": vec.scale[key], "norm": float(np.linalg.norm(v)),
                     "dose_alpha1": float(np.linalg.norm(v) / max(h_norm, EPS)),
                     "cos_coco": cos(v, vec.own["COCO"]), "cos_pool": cos(v, pool),
                     "cos_own_subset": cos(v, vec.own[sub]) if sub else math.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GPU pass
# ---------------------------------------------------------------------------

def describe(dec, low_all, iou_all, mm: int, gt, m_base):
    """One decoder pass in one output mode -> the dump's per-box numbers, and the mask."""
    import steer_models_candidates as SMC
    import steer_models_scan as SMS
    low, head = dec.pick(low_all, iou_all, bool(mm))
    m = dec.masks(low)
    ar = torch.arange(len(low), device=low.device)
    r = {"iou": SMS.batch_iou(m, gt).cpu().numpy(), "pred": iou_all[ar, head].float().cpu().numpy(),
         "stab": SMC.stability(low).cpu().numpy(), "frac": m.flatten(1).float().mean(1).cpu().numpy(),
         "chg": SMC.mask_change(m, m if m_base is None else m_base).cpu().numpy(), "head": head.cpu().numpy()}
    return r, m


def image_runs(runner, dec, site, item: dict, vecs: dict, alphas, checks: dict):
    """Unsteered + every variant x alpha on one image's boxes -> ({variant: {mm:
    {quantity: [alpha 0, alphas...]}}}, the unsteered numbers per mode)."""
    import steer_models_scan as SMS
    b, gt = item["b"], item["gt"]
    low0, iou0 = runner.forward(b)
    lz, iz = runner.forward(b, edits={site.id: SMS.zero_edit})
    z = checks["zero_dose"]
    z["forwards"] += 1
    if not (torch.equal(lz, low0) and torch.equal(iz, iou0)):
        z["ok"] = False
        z["failures"].append(f"{item['dataset']}/{item['stem']}")
        raise SMS.CheckFailed(f"zero dose at {item['dataset']}/{item['stem']}")
    z["bit_exact"] += 1
    base, m0 = {}, {}
    for mm, _ in MODES:
        base[mm], m0[mm] = describe(dec, low0, iou0, mm, gt, None)
    out = OrderedDict()
    for name, v in vecs.items():
        res = {mm: {k: [x] for k, x in base[mm].items()} for mm, _ in MODES}
        for a in alphas:
            low_all, iou_all = runner.forward(b, edits={site.id: SMS.edit_rows(a * v)})
            for mm, _ in MODES:
                r, _ = describe(dec, low_all, iou_all, mm, gt, m0[mm])
                for k, x in r.items():
                    res[mm][k].append(x)
        out[name] = res
    return out, base


def group_check(G: pd.DataFrame) -> dict:
    """build_candidate_pool.group_check: the round(n / 2) worst of an image's n
    annotations are bottom50 (12 / 13 for the 25 mobile ones)."""
    c = G.groupby(["image_name", "group"]).size().unstack(fill_value=0)
    n = c.get("bottom50", 0) + c.get("top50", 0)
    off = int((c.get("bottom50", 0) != (n * 0.5).round().astype(int)).sum())
    return {"images": int(len(c)), "n_images_off": off,
            "annotations_per_image": {int(k): int(v) for k, v in n.value_counts().sort_index().items()}, "ok": off == 0}


def run_gpu(args) -> int:
    """The GPU pass; checks.json is written however it ends."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = {"zero_dose": {"forwards": 0, "bit_exact": 0, "ok": True, "failures": []},
              "own_forward": {"compared": 0, "bit_exact": 0, "ok": True, "failures": []},
              "public_api": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-4": 0, "where": []},
              "public_single": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-3": 0, "where": []}}
    try:
        return gpu_pass(args, out, checks)
    finally:
        (out / "checks.json").write_text(json.dumps(checks, indent=1, default=_json_default), encoding="utf-8")


def gpu_pass(args, out: Path, checks: dict) -> int:
    import steer_models_candidates as SMC
    import steer_models_scan as SMS
    import steer_user_study as SUS
    from heatmaps.comp_hw_smoothed import get_original_size, load_model
    from segment_anything.utils.transforms import ResizeLongestSide
    from steer_sam_decoder import assign_folds

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    device = torch.device(args.device)
    alphas = list(args.alphas)

    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=device)
    for prm in predictor.model.parameters():
        prm.requires_grad_(False)
    # sam3 turns TF32 matmuls on at import; the scan and the pool run in fp32
    tf32 = {"matmul_after_load": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if not args.keep_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    tf32["matmul_run"] = bool(torch.backends.cuda.matmul.allow_tf32)
    checks["tf32"] = tf32
    dec = SMS.Decoder(predictor)
    if dec.training:
        raise SMS.CheckFailed("mask decoder is in training mode: the stability fallback would be off")
    site = next((s for s in SMS.build_sites(dec) if s.id == args.site), None)
    if site is None or site.kind != "tok":
        raise SystemExit(f"--site {args.site}: a token-stream site is needed (L1/L2.norm1-3, final_norm)")
    runner = SMS.Runner(dec, [site], args.batch)

    root = Path(args.root)
    datasets = list(dict.fromkeys(["COCO", *(args.datasets or SUS.DATASETS)]))
    natural = [d for d in datasets if d not in SUS.MEDICAL]
    if "pool_nat" in args.variants and not natural:
        raise SystemExit("pool_nat needs a natural subset")
    index = SUS.index_study(root, args.use, datasets)
    for d in datasets:
        if not index[d]:
            raise SystemExit(f"no user masks for {d} under {root}")
    halves = {d: SUS.split_halves(index[d].keys(), args.split_seed) for d in datasets}
    checks["split"] = {d: {"images": len(index[d]), "A": len(A), "B": len(B), "disjoint": not set(A) & set(B)}
                       for d, (A, B) in halves.items()}
    checks["split"]["ok"] = all(c["disjoint"] and c["A"] == c["images"] // 2 and c["A"] + c["B"] == c["images"]
                                for c in checks["split"].values())
    (out / "split.json").write_text(json.dumps({"seed": args.split_seed, **{d: {"A": A, "B": B} for d, (A, B)
                                                                            in halves.items()}}, indent=1),
                                    encoding="utf-8")
    if not checks["split"]["ok"]:
        raise SMS.CheckFailed(f"split: {checks['split']}")
    scan = SMC.load_scan(Path(args.scan_dir), site, dec, halves["COCO"][0], args)
    checks["scan"] = scan["check"]
    if not scan["check"]["ok"]:
        raise SMS.CheckFailed(f"--scan_dir is not this model's run: {scan['check']}")
    k, fseed = int(scan["check"]["folds"]), int(scan["check"]["seed"])
    folds_of = {d: assign_folds(halves[d][0], k, fseed) for d in datasets}
    lim = (lambda xs: list(xs)[:args.limit_images]) if args.limit_images > 0 else list
    print(f"{args.model_name}: {dec.T} tokens, fallback {dec.dynamic}, TF32 matmul {tf32['matmul_after_load']} -> "
          f"{tf32['matmul_run']}; site {site.id}; {len(datasets)} subsets, {k} folds; variants {args.variants}; "
          f"alphas {alphas}", flush=True)

    # --- pass 1: every subset's half A, unsteered, the site recorded ---------------
    dimg = {d: ([], []) for d in datasets}
    h_norms, base1 = [], {}
    for d in datasets:
        A = lim(halves[d][0])
        for i, stem in enumerate(A):
            item = SMS.load_item(dec, root, d, "A", stem, index[d], folds_of[d][stem])
            buf = []
            low0, iou0 = runner.forward(item["b"], record={site.id: lambda t, ctx, j: buf.append(t.detach().float().cpu())})
            base = SMS.score_modes(dec, low0, iou0, item["gt"])
            if i == 0:
                SMS.check_official(dec, item, low0, iou0, base, checks)
            X = torch.cat(buf).numpy()
            g = SUS.assign_groups(pd.DataFrame(item["rows"]).assign(undef_iou=base[0][0])).to_numpy()
            top, bot = g == "top50", g == "bottom50"
            if not top.any() or not bot.any():
                raise SMS.CheckFailed(f"{d}/{stem}: an empty half ({bot.sum()} bottom50, {top.sum()} top50)")
            dimg[d][0].append(stem)
            dimg[d][1].append(X[top].mean(0) - X[bot].mean(0))
            h_norms.append(np.linalg.norm(X.reshape(len(X), -1), axis=1))
            base1[(d, stem)] = {mm: base[mm][0].copy() for mm, _ in MODES}
            print(f"[A 1/2 {d}] [{i + 1}/{len(A)}] {stem}: {len(X)} boxes, undef {base[0][0].mean():.4f}, "
                  f"head-select {base[1][0].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
            del item
    dimg = {d: (stems, np.stack(x).astype(np.float32)) for d, (stems, x) in dimg.items()}
    h_norm = float(np.concatenate(h_norms).mean())

    # --- vectors ----------------------------------------------------------------------
    u = np.random.default_rng([args.seed, 202]).standard_normal((dec.T, 256)).astype(np.float32)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    try:
        vec = Vectors(dimg, folds_of, k, args.variants, natural, u, args.norm, allow_empty_fold=args.limit_images > 0)
    except RuntimeError as e:
        raise SMS.CheckFailed(str(e))
    diffs = {"all": rel_diff(vec.own["COCO"], scan["V"]["all"]),
             **{f"fold{j}": rel_diff(vec.fold[("COCO", j)], scan["V"][j]) for j in range(k)}}
    strict = (args.use == scan["meta"]["args"].get("use") and args.limit_images == 0
              and int(scan["meta"]["args"].get("limit_images", 0)) == 0)
    cv = {"rel_diff": diffs, "max_rel_diff": max(diffs.values()), "tol": args.vector_tol, "strict": strict,
          "cos_all": cos(vec.own["COCO"], scan["V"]["all"])}
    cv["ok"] = bool(cv["max_rel_diff"] <= args.vector_tol) if strict and not args.allow_vector_mismatch else None
    checks["coco_vs_scan"] = cv
    checks["empty_folds"] = vec.empty_folds
    print(f"COCO A vector vs the scan's: max rel. diff {cv['max_rel_diff']:.2e} (all + {k} folds), cos "
          f"{cv['cos_all']:.6f}{'' if strict else ' (reported only)'}", flush=True)
    if cv["ok"] is False:
        raise SMS.CheckFailed("COCO's vector is not the scan's: another model, checkpoint or site? "
                              "--allow_vector_mismatch to go on anyway")
    stats = vector_stats(vec, h_norm)
    stats.to_csv(out / "vector_stats.csv", index=False)
    npz = {}
    for d, (stems, x) in dimg.items():
        npz[f"own/{d}"], npz[f"dimg/{d}"], npz[f"stems/{d}"] = vec.own[d], x, np.asarray(stems)
        for j in range(k):
            npz[f"own/{d}/fold{j}"] = vec.fold[(d, j)]
    for key, v in vec.full().items():
        npz[f"var/{key}"] = v
    np.savez_compressed(out / "vectors.npz", **npz)
    print("vectors: " + ", ".join(f"{r.vector} x{r.scale:.2f} (cos COCO {r.cos_coco:+.2f})" for r in stats.itertuples()
                                  if not r.vector.startswith(("own:", "lodo:")) or r.vector.endswith(":COCO")),
          flush=True)

    # --- pass 2: every half of every subset, every variant x alpha --------------------------
    cache, rows = {}, []
    checks["baseline_repeat"] = {"images": 0, "max_abs_iou_diff": 0.0}
    plan = [(d, h, s) for d in datasets for h, ss in (("A", lim(halves[d][0])), ("B", lim(halves[d][1]))) for s in ss]
    la = alphas.index(min(alphas, key=lambda x: abs(x - args.log_alpha))) + 1
    t1 = time.time()
    for i, (d, half, stem) in enumerate(plan):
        fold = folds_of[d][stem] if half == "A" else -1
        key = (d, half, fold)
        if key not in cache:
            cache[key] = {n: torch.as_tensor(v, device=device) for n, v in vec.for_image(d, half, fold).items()}
        item = SMS.load_item(dec, root, d, half, stem, index[d], fold)
        res, base = image_runs(runner, dec, site, item, cache[key], alphas, checks)
        if half == "A":
            prev = base1[(d, stem)]
            dd = max(float(np.abs(prev[mm] - base[mm]["iou"]).max()) for mm, _ in MODES)
            r = checks["baseline_repeat"]
            r["images"] += 1
            r["max_abs_iou_diff"] = max(r["max_abs_iou_diff"], dd)
        g = SUS.assign_groups(pd.DataFrame(item["rows"]).assign(undef_iou=base[0]["iou"])).to_numpy()
        hw = get_original_size(predictor)
        ih, iw = ResizeLongestSide.get_preprocess_shape(hw[0], hw[1], 1024)
        bx = item["boxes"]
        bw, bh = (bx[:, 2] - bx[:, 0]).clip(min=1e-3), (bx[:, 3] - bx[:, 1]).clip(min=1e-3)
        for name, r in res.items():
            R = SMC.multimask_rows(item, r, alphas, bw * bh / float(ih * iw), np.log(bw / bh), f"{d}_{half}")
            R.insert(0, "variant", name)
            R["role"] = "train" if half == "A" else "eval"
            R["group"] = g
            R["site"] = site.id
            rows.append(R)
        shown = [n for n in res if n in ("own", "coco", "pool") or n == f"lodo:{d}"]
        print(f"[2/2 {d}_{half}] [{i + 1}/{len(plan)}] {stem}: undef {base[0]['iou'].mean():.4f} hsel "
              f"{base[1]['iou'].mean():.4f}; {_a(alphas[la - 1])} mm0/mm1 "
              + ", ".join(f"{n} {res[n][0]['iou'][la].mean():.4f}/{res[n][1]['iou'][la].mean():.4f}" for n in shown)
              + f" ({len(res)} variants, {time.time() - t1:.0f}s)", flush=True)
        del item, res

    D = pd.concat(rows, ignore_index=True)
    D.to_csv(out / "rows.csv.gz", index=False, float_format="%.6f")
    first = D.drop_duplicates(SUS.KEY)
    checks["groups"] = group_check(first)
    ref = SMS.read_reference(args.reference_csv, args.use)
    for name, col, ref_col in (("repro", "tok0_iou0", "undefended_iou"), ("repro_headsel", "multi_iou0", "headsel_iou")):
        checks[name] = SMS.repro(first, ref, col, ref_col, args.repro_tol)
    meta = {"script": "steer_universal.py", "created": time.strftime("%Y-%m-%d %H:%M"), "model_name": args.model_name,
            "checkpoint": args.checkpoint_path, "site": site.id, "tokens": "all", "use": args.use,
            "datasets": datasets, "natural": natural, "variants": list(args.variants), "alphas": alphas,
            "folds": k, "fold_seed": fseed, "norm": args.norm, "target_norm": vec.target, "h_norm": h_norm,
            "scan_dir": args.scan_dir, "images": {f"{d}_{h}": len(lim(halves[d][0 if h == "A" else 1]))
                                                 for d in datasets for h in ("A", "B")},
            "rows": int(len(D)), "boxes": int(len(first)), "layout": dec.layout(),
            "versions": SMS.package_versions(), "args": vars(args), "seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default), encoding="utf-8")
    z, of = checks["zero_dose"], checks["own_forward"]
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}: {len(first)} boxes, {len(D)} rows; zero dose "
          f"{z['bit_exact']}/{z['forwards']}, own forward {of['bit_exact']}/{of['compared']}, public API max |dIoU| "
          f"{checks['public_api']['max_abs_iou_diff']:.3g}, baseline repeat "
          f"{checks['baseline_repeat']['max_abs_iou_diff']:.3g}, groups off {checks['groups']['n_images_off']}\n"
          f"tables: python scripts/steer_universal.py --from_dump {out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Tables (CPU)
# ---------------------------------------------------------------------------

FAMILY = OrderedDict([
    ("own", "свой вектор набора (его половина A)"),
    ("coco", "вектор COCO A, α по COCO A — как в пуле и в статье сейчас"),
    ("coco_allA", "вектор COCO A, α по A всех наборов"),
    ("pool", "общий вектор: среднее векторов всех наборов"),
    ("pool_norm", "общий вектор из векторов единичной нормы"),
    ("pool_nat", "общий вектор натуральных наборов (без медицины)"),
    ("lodo", "общий вектор без своего набора (leave one dataset out): домен не виден"),
    ("rand", "случайное направление с нормами pool по токенам (контроль)"),
])
POLICY = OrderedDict([("α*", "одна α на обучающих строках"), ("GBM", "регрессор выигрыша α по боксу (steer_gate)"),
                      ("α* набора", "одна α по половине A тестового набора (доза домена)"),
                      ("оракул α", "лучшая одна α на самой тестовой половине (задним числом)")])
PAIRS = [("pool", "own"), ("pool", "coco"), ("pool", "coco_allA"), ("lodo", "coco"), ("lodo", "own"),
         ("pool", "rand"), ("pool_nat", "pool"), ("pool_norm", "pool"), ("coco_allA", "coco")]


def view(R: pd.DataFrame, pre: str, alphas) -> pd.DataFrame:
    """One mode in the column names steer_gate expects (alpha 0 = undef_iou:
    the mode's unsteered IoU)."""
    import steer_user_study as SUS
    V = R[["dataset", "half", *SUS.KEY, "group", "box_frac", "log_aspect"]].reset_index(drop=True).copy()
    V["undef_iou"] = R[f"{pre}_iou0"].to_numpy()
    for c in ("pred0", "stab0", "mask_frac0", "fill0"):
        V[c] = R[f"{pre}_{c}"].to_numpy()
    for a in alphas:
        k = _a(a)
        for c in ("iou", "pred", "chg", "fill", "stab"):
            V[f"{c}_{k}"] = R[f"{pre}_{c}_{k}"].to_numpy()
    return V


def families(present: set, datasets, natural) -> list:
    """name, the variant tested on subset d, the training rows (variant, subset
    filter of the A halves), whether the fit depends on d, the own-dose policy."""
    S = []
    has = lambda v: v in present  # noqa: E731
    if has("own"):
        S.append(dict(name="own", test=lambda d: "own", train=lambda d: ("own", [d]), per_d=True, dose=False))
    if has("coco"):
        S.append(dict(name="coco", test=lambda d: "coco", train=lambda d: ("coco", ["COCO"]), per_d=False, dose=True))
        S.append(dict(name="coco_allA", test=lambda d: "coco", train=lambda d: ("coco", list(datasets)), per_d=False,
                      dose=False))
    for v in ("pool", "pool_norm"):
        if has(v):
            S.append(dict(name=v, test=lambda d, v=v: v, train=lambda d, v=v: (v, list(datasets)), per_d=False,
                          dose=True))
    if has("pool_nat"):
        S.append(dict(name="pool_nat", test=lambda d: "pool_nat", train=lambda d: ("pool_nat", list(natural)),
                      per_d=False, dose=True))
    if any(v.startswith("lodo:") for v in present):
        S.append(dict(name="lodo", test=lambda d: f"lodo:{d}",
                      train=lambda d: (f"lodo:{d}", [e for e in datasets if e != d]), per_d=True, dose=False))
    if has("rand"):
        S.append(dict(name="rand", test=lambda d: "rand", train=lambda d: ("rand", list(datasets)), per_d=False,
                      dose=False))
    return S


def geometry(vz, datasets, natural, n_split: int, seed: int):
    """Cosines between the subsets' own vectors; per subset its split-half
    noise level (two disjoint halves of its A images) and cos with COCO, the
    pool and the leave-one-out mean."""
    own = {d: vz[f"own/{d}"] for d in datasets}
    rng = np.random.default_rng(seed)
    C = pd.DataFrame([[cos(own[a], own[b]) for b in datasets] for a in datasets], index=datasets, columns=datasets)
    pool = np.mean([own[d] for d in datasets], 0)
    nat = np.mean([own[d] for d in natural], 0) if natural else None
    rows = []
    for d in datasets:
        x = vz[f"dimg/{d}"]
        h = len(x) // 2
        sh = []
        for _ in range(n_split if h >= 1 else 0):
            p = rng.permutation(len(x))
            sh.append(cos(x[p[:h]].mean(0), x[p[h:2 * h]].mean(0)))
        r = float(np.mean(sh)) if sh else math.nan
        lodo = np.mean([own[e] for e in datasets if e != d], 0)
        rows.append({"dataset": d, "images": len(x), "norm": float(np.linalg.norm(own[d])), "split_half": r,
                     "reliability": 2 * r / (1 + r) if np.isfinite(r) and r > -1 else math.nan,
                     "cos_coco": cos(own[d], own["COCO"]), "cos_pool": cos(own[d], pool),
                     "cos_lodo": cos(own[d], lodo), "cos_pool_nat": cos(own[d], nat) if nat is not None else math.nan})
    return C, pd.DataFrame(rows)


def analyse(args) -> int:
    import steer_gate as G
    import steer_user_study as SUS
    import steer_variants as SV

    t0 = time.time()
    out = Path(args.from_dump)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    checks = json.loads((out / "checks.json").read_text(encoding="utf-8")) if (out / "checks.json").is_file() else {}
    alphas, datasets = [float(a) for a in meta["alphas"]], list(meta["datasets"])
    natural = [d for d in datasets if d not in SUS.MEDICAL]
    KEY = SUS.KEY
    D = pd.read_csv(out / "rows.csv.gz", dtype={k: str for k in KEY})
    D[KEY] = D[KEY].fillna("")
    by_var = {v: R.reset_index(drop=True) for v, R in D.groupby("variant", sort=False)}
    S = families(set(by_var), datasets, natural)
    print(f"{len(D)} rows, {len(by_var)} variants, families {[s['name'] for s in S]} ({time.time() - t0:.0f}s)",
          flush=True)

    def rows_of(v: str, subsets, half: str) -> pd.DataFrame:
        R = by_var[v]
        return R[R.dataset.isin(subsets) & (R.half == half)].reset_index(drop=True)

    fits, per, per2, curve, astar = {}, OrderedDict(), OrderedDict(), [], []
    for d in datasets:
        base, rows = None, OrderedDict()
        for mm, pre in MODES:
            for sp in S:
                tv = sp["test"](d)
                Te = rows_of(tv, [d], "B")
                if base is None:
                    base = Te
                    u = Te.tok0_iou0.to_numpy(float)
                    rows["undef"] = (u, u)
                    rows["head-select"] = (Te.multi_iou0.to_numpy(float), u)
                elif not (Te[KEY].to_numpy() == base[KEY].to_numpy()).all():
                    raise RuntimeError(f"{d}: the boxes of {tv} are not in the order of the others")
                Vte = view(Te, pre, alphas)
                fk = (sp["name"], d if sp["per_d"] else "", mm)
                if fk not in fits:
                    v, subs = sp["train"](d)
                    Vtr = view(rows_of(v, subs, "A"), pre, alphas)
                    j, a = SV.tune_alpha(Vtr, alphas)
                    fits[fk] = (j, a, G.GainRegressors(alphas, "out", args.seed).fit(Vtr), len(Vtr),
                                int(Vtr.image_name.nunique()))
                    print(f"  fit mm{mm} {sp['name']}{'/' + d if sp['per_d'] else ''}: {len(Vtr)} boxes, "
                          f"α* {a:g} ({time.time() - t0:.0f}s)", flush=True)
                j, a, reg, ntr, nimg = fits[fk]
                W = G.iou_matrix(Vte, alphas)
                lab = f"mm{mm} · {sp['name']}"
                rows[f"{lab} · α*"] = (W[:, j], u)
                rows[f"{lab} · GBM"] = (W[np.arange(len(W)), reg.choose(Vte)["gbm"]], u)
                a_d = math.nan
                if sp["dose"]:
                    jd, a_d = SV.tune_alpha(view(rows_of(tv, [d], "A"), pre, alphas), alphas)
                    rows[f"{lab} · α* набора"] = (W[:, jd], u)
                jo, a_o = SV.tune_alpha(Vte, alphas)
                rows[f"{lab} · оракул α"] = (W[:, jo], u)
                astar.append({"family": sp["name"], "mode": f"mm{mm}", "dataset": d, "alpha_train": a,
                              "train_boxes": ntr, "train_images": nimg, "alpha_own_subset": a_d, "alpha_oracle": a_o})
                if sp["name"] != "coco_allA":
                    grp = Te.group.to_numpy()
                    for jj, al in enumerate([0.0] + alphas):
                        for g in GROUPS:
                            m = np.ones(len(Te), bool) if g == "all" else grp == g
                            curve.append({"family": sp["name"], "mode": f"mm{mm}", "dataset": d, "alpha": al,
                                          "group": g, "iou": float(W[m, jj].mean()), "undef": float(u[m].mean())})
        per[d] = (base, rows)
        r2 = OrderedDict()
        names = {s["name"] for s in S}
        for mm, _ in MODES:
            for pol in ("GBM", "α*"):
                for x, y in PAIRS:
                    if x in names and y in names:
                        r2[f"mm{mm} · {x} − {y} · {pol}"] = (rows[f"mm{mm} · {x} · {pol}"][0],
                                                            rows[f"mm{mm} · {y} · {pol}"][0])
        for pol in ("GBM", "α*"):
            for x in ("pool", "lodo", "coco"):
                if x in names:
                    r2[f"mm1 · {x} − head-select · {pol}"] = (rows[f"mm1 · {x} · {pol}"][0], rows["head-select"][0])
        per2[d] = (base, r2)

    T = G.score_blocks(per, args)
    T.to_csv(out / "table.csv", index=False)
    P = G.score_blocks(per2, args)
    P.to_csv(out / "pairs.csv", index=False)
    Cv = pd.DataFrame(curve)
    Cv.to_csv(out / "alpha_curve.csv", index=False)
    As = pd.DataFrame(astar)
    As.to_csv(out / "alpha_star.csv", index=False)
    vz = np.load(out / "vectors.npz")
    C, Gm = geometry(vz, datasets, natural, args.n_split, args.seed)
    C.to_csv(out / "cos_own.csv")
    Gm.to_csv(out / "geometry.csv", index=False)
    vs = pd.read_csv(out / "vector_stats.csv") if (out / "vector_stats.csv").is_file() else None
    text = summary_md(T, P, Cv, As, C, Gm, vs, meta, checks, datasets, natural, S)
    (out / "summary.md").write_text(text, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print("\n" + text.encode(enc, "replace").decode(enc))
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------

def _means(T: pd.DataFrame, datasets) -> dict:
    """score_blocks' mean blocks: all / natural / medical -> block label (None if absent)."""
    blocks = [b for b in dict.fromkeys(T.block) if b not in datasets]
    pick = lambda p: next((b for b in blocks if b.startswith(p)), None)  # noqa: E731
    return {"all": pick("среднее по"), "nat": pick("среднее, натуральные"), "med": pick("среднее, медицина")}


def _cell(T: pd.DataFrame, block, method: str, group: str, what: str = "pct") -> str:
    if block is None:
        return "—"
    r = T[(T.block == block) & (T.method == method) & (T.group == group)]
    if not len(r):
        return "—"
    r = r.iloc[0]
    if what == "pct":
        return f"{r.pct:+.2f}%"
    if what == "iou":
        return f"{r.iou:.4f}"
    return f"{r.d:+.4f} [{r.d_lo:+.4f}, {r.d_hi:+.4f}]"


def _bold_best(cells: list, values: list) -> list:
    ok = [v for v in values if v is not None and np.isfinite(v)]
    if not ok:
        return cells
    best = max(ok)
    return [f"**{c}**" if v is not None and np.isfinite(v) and np.isclose(v, best) else c for c, v in zip(cells, values)]


def md_main(T: pd.DataFrame, methods: list, datasets) -> list:
    M = _means(T, datasets)
    cols = [("nat", "all"), ("nat", "bottom50"), ("nat", "top50"), ("med", "all"), ("med", "bottom50"),
            ("med", "top50"), ("all", "all"), ("all", "bottom50"), ("all", "top50")]
    L = ["| метод | нат. all | нат. b50 | нат. t50 | мед. all | мед. b50 | мед. t50 | все all | все b50 | все t50 "
         "| Δ IoU все all [95% CI] |", "|---|" + "---|" * 10]
    for m in methods:
        L.append(f"| {m} | " + " | ".join(_cell(T, M[b], m, g) for b, g in cols) + " | "
                 + (_cell(T, M["all"], m, "all", "ci") if m != "undef" else "—") + " |")
    return L


def summary_md(T, P, Cv, As, C, Gm, vs, meta, checks, datasets, natural, S) -> str:
    names = [s["name"] for s in S]
    alphas = meta["alphas"]
    medical = [d for d in datasets if d not in natural]
    n_img = meta.get("images", {})
    L = [f"# Универсальный вектор steering: {meta['model_name']}, узел `{meta['site']}`, разметка `{meta['use']}`", "",
         f"Каждый из {len(datasets)} наборов user study разбит на половины A / B (у COCO — половины скана). "
         "По половине A каждого набора считается свой вектор (DM внутри картинки, 13 лучших − 12 худших, все токены), "
         "из них строятся варианты ниже; **все числа — на отложенных половинах B** всех наборов "
         f"({sum(v for k, v in n_img.items() if k.endswith('_B'))} картинок). Нормировка векторов: `{meta['norm']}`"
         + (" (все к норме вектора COCO A, одна α = одна доза)" if meta["norm"] == "coco" else "") + "; "
         f"α ∈ {{{', '.join(f'{a:g}' for a in alphas)}}}. Картинки половины A стирятся вектором, где вклад их набора "
         "посчитан без их фолда, так что α и регрессор учатся на внешних эффектах.", "",
         "Ячейка — прирост IoU к undef (multimask=False, без steering) в %; среднее по наборам, каждый с равным весом. "
         "b50 / t50 — 12 худших / 13 лучших аннотаций картинки по undef. mm0 — multimask=False, mm1 — multimask "
         "с выбором головы по предсказанному IoU (head-select — его α = 0).", "",
         "**Векторы:**", ""]
    L += [f"- `{n}` — {FAMILY[n]}" for n in names]
    L += ["", "**Политики α:**", ""] + [f"- `{p}` — {t}" for p, t in POLICY.items()]
    L += ["", "Обучающие строки: `own` — A своего набора; `coco` — A COCO; `coco_allA`, `pool`, `pool_norm`, `rand` — "
          "A всех наборов; `pool_nat` — A натуральных; `lodo` — A всех наборов, кроме тестового.", ""]

    base = ["undef", "head-select"]
    L += ["## Главное: регрессор α (GBM)", ""]
    L += md_main(T, base + [f"mm{mm} · {n} · GBM" for mm, _ in MODES for n in names], datasets)
    L += ["", "## Все политики α", ""]
    pol = [f"mm{mm} · {n} · {p}" for mm, _ in MODES for n in names for p in POLICY
           if f"mm{mm} · {n} · {p}" in set(T.method)]
    L += md_main(T, base + pol, datasets)

    L += ["", "## Попарно (тот же бокс, разность IoU; 95% CI — бутстреп по картинкам внутри наборов)", "",
          "`x − y`: Δ > 0 — x лучше. % — к y.", ""]
    M = _means(P, datasets)
    L += ["| сравнение | нат. all | мед. all | все all | все b50 | все t50 |", "|---|---|---|---|---|---|"]
    for m in dict.fromkeys(P.method):
        cells = [_cell(P, M[b], m, "all", "ci") + f" ({_cell(P, M[b], m, 'all')})" if M[b] else "—"
                 for b in ("nat", "med", "all")]
        cells += [_cell(P, M["all"], m, g, "ci") for g in ("bottom50", "top50")]
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    L += ["", "## По наборам: GBM, all, % к undef (половины B)", ""]
    L += ["| метод | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    for mm, _ in MODES:
        meth = (["undef", "head-select"] if mm == 0 else []) + [f"mm{mm} · {n} · GBM" for n in names]
        for m in meth:
            vals = [T[(T.block == d) & (T.method == m) & (T.group == "all")].pct for d in datasets]
            L.append(f"| {m} | " + " | ".join(f"{v.iloc[0]:+.2f}%" if len(v) else "—" for v in vals) + " |")
    L += ["", "Лучшее направление при лучшей дозе (оракул α), all, % к undef:", "",
          "| метод | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    for mm, _ in MODES:
        for n in names:
            m = f"mm{mm} · {n} · оракул α"
            vals = [T[(T.block == d) & (T.method == m) & (T.group == "all")].pct for d in datasets]
            v2 = [v.iloc[0] if len(v) else None for v in vals]
            L.append(f"| {m} | " + " | ".join(f"{v:+.2f}%" if v is not None else "—" for v in v2) + " |")

    L += ["", "## Доза", "", "α* (обучение) / α* набора / оракул α по наборам; «—» — политики нет у семейства. "
          f"Край сетки — {max(alphas):g}.", ""]
    for mm, _ in MODES:
        A = As[As["mode"] == f"mm{mm}"]
        L += [f"**mm{mm}**", "", "| набор | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for d in datasets:
            cells = []
            for n in names:
                r = A[(A.family == n) & (A.dataset == d)]
                if not len(r):
                    cells.append("—")
                    continue
                r = r.iloc[0]
                own = "—" if not np.isfinite(r.alpha_own_subset) else f"{r.alpha_own_subset:g}"
                cells.append(f"{r.alpha_train:g} / {own} / {r.alpha_oracle:g}")
            L.append(f"| {d} | " + " | ".join(cells) + " |")
        L.append("")
    edge = As[(As.alpha_train == max(alphas)) | (As.alpha_oracle == max(alphas))]
    if len(edge):
        L += [f"На краю сетки (α = {max(alphas):g}): " + ", ".join(sorted({f"{r.family}/{r.mode}" for r in edge.itertuples()}))
              + " — доза может быть недобрана.", ""]
    L += ["Отклик на дозу: одна α для всех боксов, all, % к undef, среднее по наборам (alpha_curve.csv — по наборам "
          "и группам).", ""]
    curve_fams = [n for n in names if n != "coco_allA"]
    for mm, _ in MODES:
        for lab, subs in (("натуральные", natural), ("медицина", medical)):
            if not subs:
                continue
            Cm = Cv[(Cv["mode"] == f"mm{mm}") & (Cv.group == "all") & Cv.dataset.isin(subs)]
            g = Cm.groupby(["family", "alpha"])[["iou", "undef"]].mean()
            L += [f"**mm{mm}, {lab}** (α = 0: {'undef' if mm == 0 else 'head-select'})", "",
                  "| α | " + " | ".join(curve_fams) + " |", "|---|" + "---|" * len(curve_fams)]
            for al in [0.0] + list(alphas):
                cells = []
                for n in curve_fams:
                    if (n, al) in g.index:
                        r = g.loc[(n, al)]
                        cells.append(f"{(r.iou / r.undef - 1) * 100:+.2f}%")
                    else:
                        cells.append("—")
                L.append(f"| {al:g} | " + " | ".join(cells) + " |")
            L.append("")

    L += ["## Геометрия векторов (сырые, до нормировки)", "",
          "Косинус по всему тензору токены × 256. На диагонали — шум: косинус векторов двух непересекающихся случайных "
          "половин картинок A этого набора. Косинус двух наборов около их диагоналей — одно направление; много ниже — "
          "разные.", "", "| | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    sh = Gm.set_index("dataset").split_half
    for a in datasets:
        L.append(f"| **{a}** | " + " | ".join(f"*{sh[a]:+.2f}*" if a == b else f"{C.loc[a, b]:+.2f}" for b in datasets)
                 + " |")
    L += ["", "| набор | картинок A | ‖v‖ | шум (половины) | надёжность ≈ 2r/(1+r) | cos COCO | cos pool | "
          "cos lodo (без себя) | cos pool_nat |", "|---|---|---|---|---|---|---|---|---|"]
    for r in Gm.itertuples():
        L.append(f"| {r.dataset} | {r.images} | {r.norm:.3f} | {r.split_half:+.2f} | {r.reliability:.2f} | "
                 f"{r.cos_coco:+.2f} | {r.cos_pool:+.2f} | {r.cos_lodo:+.2f} | {r.cos_pool_nat:+.2f} |")
    if vs is not None:
        show = vs[~vs.vector.str.match(r"(own|lodo):") | vs.vector.str.endswith(":COCO")]
        L += ["", f"Векторы, как они стирятся (средняя ‖h‖ на A: {meta['h_norm']:.2f}; vector_stats.csv — все):", "",
              "| вектор | ‖v‖ сырой | множитель | доза при α = 1 | cos COCO | cos pool |", "|---|---|---|---|---|---|"]
        for r in show.itertuples():
            L.append(f"| {r.vector} | {r.raw_norm:.3f} | {r.scale:.2f} | {r.dose_alpha1:.4f} | {r.cos_coco:+.2f} | "
                     f"{r.cos_pool:+.2f} |")

    L += ["", "## Проверки", ""]
    sc, cv = checks.get("scan", {}), checks.get("coco_vs_scan", {})
    z, of, pa = checks.get("zero_dose", {}), checks.get("own_forward", {}), checks.get("public_api", {})
    br, gr = checks.get("baseline_repeat", {}), checks.get("groups", {})
    L += [f"- скан `{meta.get('scan_dir')}`: модель / layout / половина A COCO совпадают: {sc.get('ok')};",
          f"- вектор COCO A = вектор скана: max отн. разница {cv.get('max_rel_diff', math.nan):.2e} (все + фолды), "
          f"cos {cv.get('cos_all', math.nan):.6f}" + ("" if cv.get("strict") else " (только отчёт)") + ";",
          f"- нулевая доза бит в бит: {z.get('bit_exact')}/{z.get('forwards')}; свой проход = MaskDecoder.forward: "
          f"{of.get('bit_exact')}/{of.get('compared')}; публичный API max |ΔIoU| {pa.get('max_abs_iou_diff', math.nan):.3g};",
          f"- повтор baseline на A: max |ΔIoU| {br.get('max_abs_iou_diff', math.nan):.3g} ({br.get('images')} картинок); "
          f"группы round(n/2): отклонений {gr.get('n_images_off')} из {gr.get('images')};"]
    for key, lab in (("repro", "undef"), ("repro_headsel", "head-select")):
        r = checks.get(key, {})
        if r.get("available"):
            L.append(f"- {lab} vs reference csv: max |Δ| {r['max_abs']:.2e}, {r['n_over_tol']}/{r['matched']} боксов "
                     f"> {r['tol']:g}, среднее {r['mean_ours']:.4f} vs {r['mean_ref']:.4f};")
    if checks.get("empty_folds"):
        L.append(f"- пустые фолды (smoke): {', '.join(checks['empty_folds'])}.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", choices=["SAM", "SAM2.1", "SAM3"])
    p.add_argument("--checkpoint_path", default=None, help="the scan's checkpoint ('' = SAM3 from the HF cache)")
    p.add_argument("--model_type", default="vit_b", help="SAM-1 only")
    p.add_argument("--root", help="user study FOR_TEST (images/, masks/, user_masks/)")
    p.add_argument("--use", default="m", help="user mask kinds: m (mobile, the scans'), p, or mp")
    p.add_argument("--datasets", nargs="*", default=None, help="subsets (default: all ten; COCO is always in)")
    p.add_argument("--scan_dir", help="steer_models_scan.py run of this model (COCO A vector, folds, split)")
    p.add_argument("--site", default="L2.norm3", help="a token-stream site of the scan")
    p.add_argument("--variants", default=",".join(VARIANTS), help=f"comma list of {', '.join(VARIANTS)}")
    p.add_argument("--alphas", type=_floats, default=_floats("1,2,3,4,6,8,12,16,24,32"))
    p.add_argument("--norm", choices=["coco", "raw"], default="coco",
                   help="coco: every vector rescaled to the COCO A vector's norm; raw: their own norms")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="the random direction, the regressors, the bootstrap")
    p.add_argument("--batch", type=int, default=64, help="boxes per decoder pass (the scan's)")
    p.add_argument("--vector_tol", type=float, default=1e-4, help="max relative |COCO vector - the scan's|")
    p.add_argument("--allow_vector_mismatch", action="store_true", help="record a failed vector check and go on")
    p.add_argument("--keep_tf32", action="store_true", help="leave TF32 matmuls as the loader set them")
    p.add_argument("--reference_csv", default="", help="per-case results of the same model (reported only)")
    p.add_argument("--repro_tol", type=float, default=1e-3)
    p.add_argument("--limit_images", type=int, default=0, help="per half and subset, for a smoke run; 0 = all")
    p.add_argument("--log_alpha", type=float, default=8.0, help="the alpha shown in the per-image log line")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir")
    p.add_argument("--from_dump", default="", help="an --out_dir of the GPU pass: the tables only, CPU")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--n_split", type=int, default=200, help="random half splits for the vectors' noise level")
    args = p.parse_args(argv)
    if not args.from_dump:
        miss = [f"--{k}" for k in ("model_name", "root", "scan_dir", "out_dir") if not getattr(args, k)]
        if args.checkpoint_path is None:
            miss.append("--checkpoint_path")
        if miss:
            p.error(f"the GPU pass needs {', '.join(miss)} (or --from_dump)")
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in args.variants if v not in VARIANTS]
    if bad:
        p.error(f"unknown variants {bad}; choose from {VARIANTS}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.from_dump:
        return analyse(args)
    # SAM-3 lives in its own conda env; re-execs there and exits (no-op otherwise)
    maybe_dispatch_to_env(args.model_name, __file__)
    import steer_models_scan as SMS
    try:
        return run_gpu(args)
    except SMS.CheckFailed as e:
        print(f"[check failed] {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
