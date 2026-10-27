"""
steer_multimask.py
==================
Steering with SAM's multimask output, against head-select as the baseline.

grad (grad_results) runs SAM-1 with multimask: at every step the mask is the
head of three with the highest predicted IoU, and step 0 -- head-select on the
user's box -- already gives most of grad's gain over the single output token
0. Steering so far used token 0 only. Here the same steering is run with
multimask on and the head picked the same way, so it can be measured against
head-select directly. The steering vector needs no change: the decoder's
token stream at L2.norm2 is the same with multimask on or off (multimask only
chooses which output tokens become masks).

PER BOX  (every user-study box, --use m, every --datasets subset)
  token 0  multimask off: undef (alpha 0) and steered at every alpha of --alphas
  multi    multimask on, head = argmax of the predicted IoU: headsel (alpha 0)
           and steered at every alpha
  both modes with steer_gate's features (predicted IoU, stability, box fill,
  what each alpha does to the mask and to the predicted IoU)
  vector   the COCO A vector of steer_user_study (--vector), at L2.norm2, all
           7 tokens; the COCO A images (the training boxes) steered with the
           vector of the other folds of A (cross-fitting, as steer_gate)

CHOOSING ALPHA  (fitted on the COCO A boxes only, per mode, as steer_variants)
  fixed alphas (--fixed_alphas), alpha* (one alpha, the best mean gain), and the
  gain regressors (linear, GBM). For multi the gain is over headsel, for token 0
  over undef. The multi oracle (best alpha per box by its GT IoU, 0 included)
  is the ceiling.

TABLES  (every test set, natural / medical / all: each set weighted equally)
  undef, headsel (baseline), grad (--reference_csv: the multimask ascent),
  steering on token 0 (GBM regressor), steering with multimask (every alpha
  method); IoU and % to undef per group (all / bottom50 / top50, inside each
  image by undef as everywhere else), and the gain over headsel in % of it with
  a paired 95% image-bootstrap CI. Also: how often the steered multimask pick
  changes the head headsel chose.

CHECKS (checks.json; the first three stop the run)
  zero dose: a zero vector reproduces both modes bit for bit on every image;
  the vector is the one of its run (split, sha256); token-0 undef vs
  --reference_csv (as steer_user_study); headsel vs the csv's headsel_iou
  (reported: grad_results computed it through the ascent's own forward);
  12 / 13 groups.

    CUDA_VISIBLE_DEVICES=0 python scripts/steer_multimask.py \\
        --root /.../FOR_TEST --checkpoint_path /.../sam_vit_b_01ec64.pth \\
        --vector exp_res/steer_user_study/vector.npz --reference_csv results/samfull_m.csv \\
        --out_dir exp_res/steer_multimask

    # the tables again from the dump, no GPU
    python scripts/steer_multimask.py --from_dump exp_res/steer_multimask/rows.csv.gz \\
        --reference_csv results/samfull_m.csv --out_dir exp_res/steer_multimask
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

import steer_gate as G  # noqa: E402
import steer_user_study as SUS  # noqa: E402
import steer_variants as SV  # noqa: E402
from steer_sam_decoder import Cfg, Ctx, Runner, assign_folds, batch_iou, build_sites, make_edit, mask_change, zero_edit  # noqa: E402

KEY, _a = SUS.KEY, SUS._a
MEDICAL = SUS.MEDICAL
GROUPS = ("all", "bottom50", "top50")
MODES = ("tok0", "multi")


class MMRunner(Runner):
    """Runner whose forward can also return SAM's three multimask outputs."""

    @torch.inference_mode()
    def forward_mm(self, boxes: torch.Tensor, edits=None):
        """-> masks (N, 3, H, W) bool, low-res logits (N, 3, h, w), predicted IoU (N, 3)."""
        edits = edits or {}
        holder = {}
        handles = [self.mods[sid].register_forward_hook(self._hook(sid, edits[sid], None, holder)) for sid in edits]
        masks, lows, preds = [], [], []
        try:
            for s in range(0, len(boxes), self.batch):
                sl = slice(s, min(s + self.batch, len(boxes)))
                holder["ctx"] = Ctx(self, boxes, sl)
                m, p, low = self.p.predict_torch(None, None, boxes=boxes[sl], multimask_output=True)
                masks.append(m)
                preds.append(p.float())
                lows.append(low)
        finally:
            for h in handles:
                h.remove()
        return torch.cat(masks), torch.cat(lows), torch.cat(preds)


def head_select(masks, lows, preds):
    """The head with the highest predicted IoU -> (mask, logits, predicted IoU, head)."""
    h = preds.argmax(1)
    ar = torch.arange(len(h), device=h.device)
    return masks[ar, h], lows[ar, h], preds[ar, h], h


def describe(prefix: str, m0, low0, p0, runs, gt, box_frac, alphas) -> dict:
    """steer_gate's per-box columns for one mode: alpha 0 and every alpha."""
    frac0 = m0.flatten(1).float().mean(1).cpu().numpy()
    c = {f"{prefix}_iou0": batch_iou(m0, gt).cpu().numpy(), f"{prefix}_pred0": p0.cpu().numpy(),
         f"{prefix}_stab0": G.stability(low0).cpu().numpy(), f"{prefix}_mask_frac0": frac0,
         f"{prefix}_fill0": frac0 / box_frac}
    for a, (m, low, p) in zip(alphas, runs):
        k = _a(a)
        c[f"{prefix}_iou_{k}"] = batch_iou(m, gt).cpu().numpy()
        c[f"{prefix}_pred_{k}"] = p.cpu().numpy()
        c[f"{prefix}_chg_{k}"] = mask_change(m, m0).cpu().numpy()
        c[f"{prefix}_fill_{k}"] = m.flatten(1).float().mean(1).cpu().numpy() / box_frac
        c[f"{prefix}_stab_{k}"] = G.stability(low).cpu().numpy()
    return c


def image_rows(runner: MMRunner, site, root: Path, table: str, ds: str, stem: str, recs, v: np.ndarray, role: str,
               args, checks: dict) -> pd.DataFrame:
    dev = runner.device
    cases, boxes, gt = SUS.load_image_cases(runner.p, root, stem, recs, dev)
    b = torch.as_tensor(boxes, device=dev)
    Vt = {site.id: {"dm": torch.as_tensor(v, device=dev)}}
    edit = {a: {site.id: make_edit(Cfg(-1, "steer", site.id, args.tokens, "dm", a), site, Vt)} for a in args.alphas}
    in_h, in_w = runner.p.input_size
    bw, bh = (boxes[:, 2] - boxes[:, 0]).clip(min=1e-3), (boxes[:, 3] - boxes[:, 1]).clip(min=1e-3)
    box_frac = bw * bh / float(in_h * in_w)

    # token 0 (multimask off)
    with G.PredHook(runner.sam) as ph:
        m0, low0 = runner.forward(b)
    p0 = ph.result()
    SUS.check_zero(runner, b, m0, low0, zero_edit, site.id, checks, f"tok0/{table}/{stem}")
    runs = []
    for a in args.alphas:
        with G.PredHook(runner.sam) as ph:
            m, low = runner.forward(b, edits=edit[a])
        runs.append((m, low, ph.result()))
    cols = describe("tok0", m0, low0, p0, runs, gt, box_frac, args.alphas)

    # multimask: the head with the highest predicted IoU
    M0, L0, P0 = runner.forward_mm(b)
    Mz, Lz, Pz = runner.forward_mm(b, edits={site.id: zero_edit})
    z = checks["zero_dose_multi"]
    z["forwards"] += 1
    if torch.equal(M0, Mz) and torch.equal(L0, Lz) and torch.equal(P0, Pz):
        z["bit_exact"] += 1
    else:
        z["failures"].append(f"{table}/{stem}")
        raise SUS.CheckFailed(f"zero-dose check failed with multimask at {table}/{stem}")
    hm0, hl0, hp0, h0 = head_select(M0, L0, P0)
    runs, heads = [], {}
    for a in args.alphas:
        hm, hl, hp, h = head_select(*runner.forward_mm(b, edits=edit[a]))
        runs.append((hm, hl, hp))
        heads[f"multi_head_{_a(a)}"] = h.cpu().numpy()
    cols.update(describe("multi", hm0, hl0, hp0, runs, gt, box_frac, args.alphas))
    cols["multi_head0"] = h0.cpu().numpy()
    cols.update(heads)
    return pd.DataFrame({"table": table, "dataset": ds, "image_name": stem, "role": role,
                         "kind": [c["kind"] for c in cases], "user": [c["user"] for c in cases],
                         "attempt": [c["attempt"] for c in cases], "box_frac": box_frac,
                         "log_aspect": np.log(bw / bh), **cols})


def gpu_pass(args, checks: dict) -> pd.DataFrame | None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(args.root)
    datasets = list(dict.fromkeys(["COCO", *args.datasets]))
    index = SUS.index_study(root, args.use, datasets)
    A, B = SUS.split_halves(index["COCO"], args.split_seed)
    device = torch.device(args.device)
    predictor = SUS.load_predictor(args.checkpoint_path, args.model_type, device)
    sites = build_sites(len(predictor.model.mask_decoder.transformer.layers))
    site = next(s for s in sites if s.id == args.site)
    runner = MMRunner(predictor, sites, args.batch, field_grid=64, field_margin=0.25)

    got = SUS.load_vector(Path(args.vector), site.id, A, checks)
    if got is None:
        return None
    v = got[0]
    z = np.load(args.vector)
    images_A, d_img = [str(s) for s in z["images_A"]], z["d_img"]
    folds = assign_folds(images_A, args.folds, args.seed)
    fold_of = np.array([folds[s] for s in images_A])
    v_fold = {k: d_img[fold_of != k].mean(0) for k in sorted(set(fold_of))}
    checks["cross_fit"] = {"folds": len(v_fold), "images_A_match_split": images_A == A,
                           "d_img_mean_is_v": bool(np.allclose(d_img.mean(0), v, atol=1e-7))}
    checks["cross_fit"]["ok"] = checks["cross_fit"]["images_A_match_split"] and checks["cross_fit"]["d_img_mean_is_v"]
    if not checks["cross_fit"]["ok"]:
        return None

    lim = (lambda s: s[:args.limit_images]) if args.limit_images > 0 else (lambda s: s)
    tables = [("COCO_A", "COCO", lim(A)), ("COCO_B", "COCO", lim(B))]
    tables += [(d, d, lim(list(index[d]))) for d in datasets if d != "COCO"]
    parts, t0 = [], time.time()
    for table, ds, stems in tables:
        print(f"\n[{table}] {len(stems)} images, alphas {', '.join(f'{a:g}' for a in args.alphas)}", flush=True)
        for i, stem in enumerate(stems):
            train = table == "COCO_A"
            vv = v_fold[folds[stem]] if train else v
            df = image_rows(runner, site, root, table, ds, stem, index[ds][stem], vv, "train" if train else "eval",
                            args, checks)
            parts.append(df)
            print(f"[{table}] [{i + 1}/{len(stems)}] {stem}: {len(df)} boxes, undef {df.tok0_iou0.mean():.4f}, "
                  f"headsel {df.multi_iou0.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    D = pd.concat(parts, ignore_index=True)
    D["group"] = SUS.assign_groups(D.rename(columns={"tok0_iou0": "undef_iou"})).to_numpy()
    return D


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def view(D: pd.DataFrame, mode: str, alphas) -> pd.DataFrame:
    """One mode in the column names steer_gate / steer_variants expect (alpha 0 = undef_iou)."""
    V = D[["table", "dataset", *KEY, "role", "group", "box_frac", "log_aspect"]].copy()
    for c in ("pred0", "stab0", "mask_frac0", "fill0"):
        V[c] = D[f"{mode}_{c}"].to_numpy()
    V["undef_iou"] = D[f"{mode}_iou0"].to_numpy()
    for a in alphas:
        k = _a(a)
        for c in ("iou", "pred", "chg", "fill", "stab"):
            V[f"{c}_{k}"] = D[f"{mode}_{c}_{k}"].to_numpy()
    return V


def per_box(D: pd.DataFrame, ref, args):
    """table -> boxes with a column per method; the multimask picks' heads."""
    alphas, fixed = list(args.alphas), [a for a in args.fixed_alphas if a in args.alphas]
    fitted, a_star = {}, {}
    for mode in MODES:
        Vtr = view(D[D.role == "train"].reset_index(drop=True), mode, alphas)
        fitted[mode], a_star[mode] = SV.variant_methods(Vtr, alphas, fixed, args.seed)
    per, heads = OrderedDict(), []
    for t in [t for t in dict.fromkeys(D.table) if t != "COCO_A"]:
        Dt = D[D.table == t].reset_index(drop=True)
        P = Dt[["image_name", "group"]].copy()
        P["undef"] = Dt.tok0_iou0.to_numpy()
        P["headsel"] = Dt.multi_iou0.to_numpy()
        if ref is not None:
            P["grad"] = Dt[KEY].merge(ref, on=KEY, how="left").grad_final_iou.to_numpy()
        V0 = view(Dt, "tok0", alphas)
        W0 = G.iou_matrix(V0, alphas)
        P["steer_tok0_gbm"] = W0[np.arange(len(V0)), fitted["tok0"](V0)["регрессор α, GBM"]]
        Vm = view(Dt, "multi", alphas)
        Wm = G.iou_matrix(Vm, alphas)
        H = Dt[["multi_head0"] + [f"multi_head_{_a(a)}" for a in alphas]].to_numpy()
        for name, c in fitted["multi"](Vm).items():
            col = f"multi: {name}"
            P[col] = Wm[np.arange(len(Vm)), c]
            heads.append({"table": t, "method": col, "steered": float((c > 0).mean()),
                          "head_changed": float((H[np.arange(len(Vm)), c] != H[:, 0]).mean())})
        P["multi: оракул α"] = Wm.max(1)
        per[t] = P
    return per, pd.DataFrame(heads), a_star


def boot_means(cols: dict, img: np.ndarray, n_boot: int, seed: int) -> dict:
    codes, inv = np.unique(img, return_inverse=True)
    N = np.bincount(inv, minlength=len(codes)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(codes), size=(n_boot, len(codes)))
    out = {}
    for k, x in cols.items():
        S = np.bincount(inv, weights=x, minlength=len(codes))
        out[k] = S[idx].sum(1) / N[idx].sum(1)
    return out


def score(per: "OrderedDict[str, pd.DataFrame]", args) -> pd.DataFrame:
    """Per block, method and group: IoU, % to undef, gain over headsel in % of it
    with a paired 95% image-bootstrap CI; then the means over the blocks (each
    weighted equally, bootstrapped within each)."""
    methods = [c for c in per[next(iter(per))].columns if c not in ("image_name", "group")]
    rows, boots = [], {}

    def row(block, g, n, m, b):
        out = []
        for c in methods:
            r = {"block": block, "group": g, "n": n, "method": c, "iou": m[c], "pct": (m[c] / m["undef"] - 1) * 100}
            if c != "headsel" and np.isfinite(m[c]):
                r["gain"] = (m[c] / m["headsel"] - 1) * 100
                r["lo"], r["hi"] = np.percentile((b[c] / b["headsel"] - 1) * 100, [2.5, 97.5])
            out.append(r)
        return out

    for ti, (t, P) in enumerate(per.items()):
        for g in GROUPS:
            s = P if g == "all" else P[P.group == g]
            ok = {c: s[c].to_numpy(float) for c in methods if s[c].notna().all()}
            b = boot_means(ok, s.image_name.to_numpy(), args.n_boot, args.seed + ti)
            boots[(t, g)] = b
            rows += row(t, g, len(s), {c: s[c].mean() for c in methods}, b)
    T = pd.DataFrame(rows)
    nat = [t for t in per if t not in MEDICAL]
    med = [t for t in per if t in MEDICAL]
    for label, ts in ((f"среднее, натуральные ({len(nat)})", nat), (f"среднее, медицина ({len(med)})", med),
                      (f"среднее по {len(per)}", list(per))):
        if len(ts) < 2:
            continue
        for g in GROUPS:
            s = T[T.block.isin(ts) & (T.group == g)]
            m = s.groupby("method").iou.mean().to_dict()
            common = set.intersection(*[set(boots[(t, g)]) for t in ts])
            b = {c: np.mean([boots[(t, g)][c] for t in ts], axis=0) for c in common}
            rows_ = row(label, g, int(s[s.method == "undef"].n.sum()), {c: m.get(c, np.nan) for c in methods}, b)
            T = pd.concat([T, pd.DataFrame(rows_)], ignore_index=True)
    return T


LABEL = {"undef": "undef (token 0)", "headsel": "headsel (multimask, бейзлайн)", "grad": "grad (multimask-подъём)",
         "steer_tok0_gbm": "steering, token 0, GBM-регрессор"}


def label(m: str) -> str:
    if m in LABEL:
        return LABEL[m]
    return m.replace("multi: ", "steering + multimask: ")


def md(T: pd.DataFrame, heads: pd.DataFrame, a_star: dict, D: pd.DataFrame, args, checks: dict) -> str:
    L = ["# Steering с multimask против headsel (SAM-1 vit_b, user study `m`)", "",
         "Steering — вектор COCO A (13 лучших − 12 худших, 25 картинок) на `L2.norm2`, все 7 токенов; маска — "
         "multimask-голова с максимальным предсказанным IoU (как headsel), бейзлайн — headsel без steering. α "
         f"выбирается на боксах COCO A (кросс-фит): фиксированные, одна α* (multimask: {a_star['multi']:g}; token 0: "
         f"{a_star['tok0']:g}), регрессоры выигрыша к headsel (линейный, GBM); оракул α — потолок. grad — "
         "multimask-подъём из `samfull_m.csv` на тех же боксах. Группы — внутри картинки по undef (token 0), "
         "12 / 13. Ячейка — IoU (% к undef); «к headsel» — прирост в % от headsel, парный бутстреп по картинкам.", ""]
    focus = ["grad", "steer_tok0_gbm", "multi: регрессор α, GBM", "multi: α* (подобрана на обучении)"]
    L += ["## Коротко: прирост к headsel, bottom50 / top50 / all", "",
          "| набор | " + " | ".join(label(m) for m in focus) + " |", "|---|" + "---|" * len(focus)]
    for block in dict.fromkeys(T.block):
        s = T[T.block == block]
        cells = []
        for m in focus:
            x = s[s.method == m].set_index("group")
            if x.empty or "gain" not in x or x.gain.isna().all():
                cells.append("—")
                continue
            cells.append(" / ".join(f"{x.loc[g, 'gain']:+.2f}" + ("*" if x.loc[g, "lo"] > 0 or x.loc[g, "hi"] < 0 else "")
                                    for g in ("bottom50", "top50", "all")))
        bold = "**" if block.startswith("среднее") else ""
        L.append(f"| {bold}{block}{bold} | " + " | ".join(cells) + " |")
    L += ["", "\\* — 95% CI не захватывает 0.", ""]
    for block in dict.fromkeys(T.block):
        s = T[T.block == block]
        L += [f"## {block}", "", "| метод | all | bottom50 | top50 | к headsel, b50 [95% CI] | к headsel, t50 [95% CI] |",
              "|---|---|---|---|---|---|"]
        for m in dict.fromkeys(s.method):
            x = s[s.method == m].set_index("group")
            cells = [f"{x.loc[g, 'iou']:.4f}" if m == "undef" else f"{x.loc[g, 'iou']:.4f} ({x.loc[g, 'pct']:+.2f}%)"
                     for g in GROUPS]
            gains = []
            for g in ("bottom50", "top50"):
                r = x.loc[g]
                gains.append("—" if m == "headsel" or not np.isfinite(r.get("gain", np.nan))
                             else f"{r.gain:+.2f}% [{r.lo:+.2f}, {r.hi:+.2f}]")
            L.append(f"| {label(m)} | " + " | ".join(cells) + f" | {gains[0]} | {gains[1]} |")
        L.append("")
    if len(heads):
        h = heads.groupby("method")[["steered", "head_changed"]].mean()
        L += ["## Как часто steering меняет голову", "", "| метод | доля боксов со steering | доля боксов, где голова "
              "не та, что у headsel |", "|---|---|---|"]
        for m, r in h.iterrows():
            L.append(f"| {label(m)} | {r.steered:.1%} | {r.head_changed:.1%} |")
        L.append("")
    z, zm = checks.get("zero_dose", {}), checks.get("zero_dose_multi", {})
    rp, hs = checks.get("repro", {}), checks.get("repro_headsel", {})
    L += ["## Проверки", "",
          f"- нулевая доза: token 0 {z.get('bit_exact', '-')}/{z.get('forwards', '-')}, multimask "
          f"{zm.get('bit_exact', '-')}/{zm.get('forwards', '-')} бит в бит;",
          f"- undef vs reference csv: max |Δ| {rp.get('max_abs_diff', math.nan):.2e}, доля > {args.repro_tol:g}: "
          f"{rp.get('frac_over', math.nan):.2%};",
          f"- headsel vs reference csv (headsel_iou): max |Δ| {hs.get('max_abs_diff', math.nan):.2e}, доля > "
          f"{args.repro_tol:g}: {hs.get('frac_over', math.nan):.2%};",
          f"- группы 12/13: отклонений {checks.get('groups', {}).get('n_images_not_12_13', '-')}."]
    return "\n".join(L) + "\n"


def check_headsel(D: pd.DataFrame, ref, args) -> dict:
    if ref is None:
        return {"status": "skipped"}
    m = D[KEY + ["multi_iou0"]].merge(ref[KEY + ["headsel_iou"]], on=KEY, how="left")
    d = (m.multi_iou0 - m.headsel_iou).abs()
    return {"boxes": len(m), "missing": int(m.headsel_iou.isna().sum()), "max_abs_diff": float(d.max()),
            "frac_over": float((d > args.repro_tol).mean()),
            "worst": m.assign(d=d).nlargest(5, "d")[KEY + ["multi_iou0", "headsel_iou", "d"]].to_dict("records")}


def run(args) -> int:
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = {"zero_dose": {"forwards": 0, "bit_exact": 0, "failures": [], "ok": True},
              "zero_dose_multi": {"forwards": 0, "bit_exact": 0, "failures": [], "ok": True}}

    def write_checks():
        (out / "checks.json").write_text(json.dumps(checks, indent=2, default=SUS._json_default), encoding="utf-8")

    ref = SUS.read_reference(args.reference_csv, args.use)
    if args.from_dump:
        D = pd.read_csv(args.from_dump, dtype={k: str for k in KEY})
        prev = Path(args.from_dump).parent / "checks.json"
        if prev.is_file():
            checks = json.loads(prev.read_text(encoding="utf-8"))
    else:
        try:
            D = gpu_pass(args, checks)
        except SUS.CheckFailed as e:
            write_checks()
            print(f"\nCHECK FAILED: {e} -- see {out / 'checks.json'}", file=sys.stderr)
            return 1
        if D is None:
            write_checks()
            print(f"\nCHECK FAILED: vector / cross-fit -- see {out / 'checks.json'}", file=sys.stderr)
            return 1
        D.to_csv(out / "rows.csv.gz", index=False)
        base = D.rename(columns={"tok0_iou0": "undef_iou"})
        checks["repro"] = SUS.check_repro(base, ref, args)
        checks["repro_headsel"] = check_headsel(D, ref, args)
        checks["groups"] = SUS.check_groups(base, ref, args, args.limit_images <= 0)
        write_checks()
        failed = [k for k, r in checks.items() if isinstance(r, dict) and r.get("ok") is False]
        if failed:
            print(f"\nCHECK FAILED: {', '.join(failed)} -- see {out / 'checks.json'}", file=sys.stderr)
            return 1
        (out / "meta.json").write_text(json.dumps({"args": vars(args), "rows": int(len(D))}, indent=2,
                                                  default=SUS._json_default), encoding="utf-8")

    per, heads, a_star = per_box(D, ref, args)
    T = score(per, args)
    T.to_csv(out / "table.csv", index=False)
    heads.to_csv(out / "heads.csv", index=False)
    text = md(T, heads, a_star, D, args, checks)
    (out / "summary.md").write_text(text, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print("\n" + text.encode(enc, "replace").decode(enc))
    meta = {"args": vars(args), "rows": int(len(D)), "alpha_star": a_star,
            "minutes": round((time.time() - t0) / 60, 1), "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=SUS._json_default), encoding="utf-8")
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


def parse_args(argv=None):
    fl = SUS._floats
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="user study FOR_TEST")
    p.add_argument("--use", default="m")
    p.add_argument("--datasets", nargs="*", default=[d for d in SUS.DATASETS if d != "COCO"],
                   choices=[d for d in SUS.DATASETS if d != "COCO"], help="test subsets besides COCO half B")
    p.add_argument("--checkpoint_path", default="")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--site", default="L2.norm2")
    p.add_argument("--tokens", default="all")
    p.add_argument("--vector", default="exp_res/steer_user_study/vector.npz")
    p.add_argument("--alphas", type=fl, default=fl("1,2,3,4,6,8,12,16"))
    p.add_argument("--fixed_alphas", type=fl, default=fl("1,4,12"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference_csv", default="")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--repro_tol", type=float, default=1e-3)
    p.add_argument("--repro_max_frac", type=float, default=0.01)
    p.add_argument("--group_tol", type=float, default=1e-3)
    p.add_argument("--limit_images", type=int, default=0, help="per table, for a smoke run")
    p.add_argument("--from_dump", default="", help="rows.csv.gz of an earlier run: tables only, no GPU")
    p.add_argument("--out_dir", default="exp_res/steer_multimask")
    args = p.parse_args(argv)
    if not args.from_dump and not args.root:
        p.error("--root is required (or --from_dump)")
    return args


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
