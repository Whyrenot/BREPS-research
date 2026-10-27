"""
steer_full_vectors.py
=====================
Is there one steering direction, or one per dataset? The vector of every
user-study dataset estimated on the whole dataset, and how alike they are.
Nothing is steered: one forward per image, the states only.

PER IMAGE  (SAM-1, L2.norm2, the 7 tokens x 256; --use m)
  the 25 user boxes as steer_user_study reads them, and the tight bbox of the
  GT mask (best_box of build_user_cases); groups bottom50 / top50 inside the
  image by undefended IoU (12 / 13)

VECTORS PER DATASET  (the mean over its images of a per-image difference)
  halves       mean(top50) - mean(bottom50)       the reported recipe
  tight_bot    h(tight GT box) - mean(bottom50)
  tight_worst  h(tight GT box) - h(the worst user box)
  tight_all    h(tight GT box) - mean(all 25)
  COCO_A25 (the 25 images of half A: the reported vector) rides along.

AGREEMENT
  cos        cosine of two datasets' vectors over the whole 7 x 256 tensor, and
             per token
  noise      cosine of the vectors of two disjoint random halves of one dataset
             (25 vs 25 images; and 10 vs 10, the size of a dataset's own vector
             in steer_variants), mean over --splits draws: how alike two
             estimates of the SAME direction are. A cross-dataset cosine near it
             means the same direction; far below it, a different one.
  corrected  cos / sqrt(r_i r_j), r = the split-half cosine stepped up to the
             full dataset (Spearman-Brown, 2r / (1 + r)): an approximate
             noise-free cosine (an analogy to correlation, not exact for cosines)

CHECKS (checks.json): the half-A part of COCO reproduces --vector (same
states, same rule); undefended IoU vs --reference_csv; 12 / 13 groups.

    CUDA_VISIBLE_DEVICES=0 python scripts/steer_full_vectors.py \\
        --root /.../FOR_TEST --checkpoint_path /.../sam_vit_b_01ec64.pth \\
        --vector exp_res/steer_user_study/vector.npz --reference_csv results/samfull_m.csv \\
        --out_dir exp_res/steer_full_vectors
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
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import steer_user_study as SUS  # noqa: E402
from steer_sam_decoder import Runner, batch_iou, build_sites, feature_of  # noqa: E402

KEY = SUS.KEY
TOKENS = SUS.TOKEN_NAMES
RULES = OrderedDict([("halves", "13 лучших − 12 худших"), ("tight_bot", "tight bbox GT − 12 худших"),
                     ("tight_worst", "tight bbox GT − 1 худшая"), ("tight_all", "tight bbox GT − все 25")])


def image_state(runner: Runner, site, root: Path, stem: str, recs, device):
    cases, boxes, gt = SUS.load_image_cases(runner.p, root, stem, recs, device)
    buf = []
    rec = {site.id: lambda t, ctx: buf.append(feature_of(site, t, ctx).cpu())}
    m0, _ = runner.forward(torch.as_tensor(boxes, device=device), record=rec)
    H = torch.cat(buf).numpy()
    buf.clear()
    mt, _ = runner.forward(torch.as_tensor(np.asarray(cases[0]["best_box"], np.float32)[None], device=device),
                           record=rec)
    df = pd.DataFrame({"image_name": stem, "kind": [c["kind"] for c in cases], "user": [c["user"] for c in cases],
                       "attempt": [c["attempt"] for c in cases], "undef_iou": batch_iou(m0, gt).cpu().numpy()})
    df["group"] = SUS.assign_groups(df).to_numpy()
    top, bot = (df.group == "top50").to_numpy(), (df.group == "bottom50").to_numpy()
    case = df[KEY].astype(str).agg("|".join, axis=1).to_numpy()
    worst = np.lexsort((case, df.undef_iou.to_numpy()))[0]
    agg = {"top": H[top].mean(0), "bot": H[bot].mean(0), "all": H.mean(0), "tight": torch.cat(buf).numpy()[0],
           "worst": H[worst]}
    return agg, df, float(batch_iou(mt, gt)[0])


def per_image_d(agg: dict) -> dict:
    return {"halves": agg["top"] - agg["bot"], "tight_bot": agg["tight"] - agg["bot"],
            "tight_worst": agg["tight"] - agg["worst"], "tight_all": agg["tight"] - agg["all"]}


def cos(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def split_cos(d: np.ndarray, n: int, splits: int, rng) -> float:
    """Mean cosine of the vectors of two disjoint random n-image subsets."""
    if 2 * n > len(d):
        return np.nan
    out = []
    for _ in range(splits):
        p = rng.permutation(len(d))
        out.append(cos(d[p[:n]].mean(0), d[p[n:2 * n]].mean(0)))
    return float(np.mean(out))


def analyse(d_img: dict, names: list, args):
    """d_img: rule -> dataset -> (n_images, 7, 256). -> tables."""
    rng = np.random.default_rng(args.seed)
    V = {r: {ds: d.mean(0) for ds, d in by.items()} for r, by in d_img.items()}
    rows, noise = {}, []
    for r in RULES:
        M = pd.DataFrame([[cos(V[r][a], V[r][b]) for b in names] for a in names], index=names, columns=names)
        rows[r] = M
        for ds in names:
            if ds == "COCO_A25":
                continue
            d = d_img[r][ds]
            half = len(d) // 2
            r_half = split_cos(d, half, args.splits, rng)
            noise.append({"rule": r, "dataset": ds, "n_images": len(d), "split_half": r_half,
                          "split_10": split_cos(d, 10, args.splits, rng),
                          "reliability_full": 2 * r_half / (1 + r_half) if np.isfinite(r_half) else np.nan,
                          "norm": float(np.linalg.norm(V[r][ds]))})
    N = pd.DataFrame(noise)
    corrected = {}
    for r, M in rows.items():
        rel = N[N.rule == r].set_index("dataset").reliability_full
        ds = [d for d in M.index if d in rel.index]
        R = M.loc[ds, ds].copy()
        for a in ds:
            for b in ds:
                if a != b:
                    R.loc[a, b] = M.loc[a, b] / np.sqrt(max(rel[a], 1e-6) * max(rel[b], 1e-6))
        corrected[r] = R
    tok = pd.DataFrame({ds: [cos(V["halves"][ds][i], V["halves"]["COCO"][i]) for i in range(len(TOKENS))]
                        for ds in names}, index=TOKENS)
    return V, rows, corrected, N, tok


def md_matrix(M: pd.DataFrame, diag: pd.Series | None = None) -> str:
    cols = list(M.columns)
    L = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for a in M.index:
        cells = []
        for b in cols:
            if a == b:
                cells.append(f"*{diag[a]:.2f}*" if diag is not None and a in diag.index else "1")
            else:
                cells.append(f"{M.loc[a, b]:+.2f}")
        L.append(f"| **{a}** | " + " | ".join(cells) + " |")
    return "\n".join(L)


def summary(rows, corrected, N, tok, args, checks) -> str:
    h = N[N.rule == "halves"].set_index("dataset")
    L = ["# Векторы steering по полным наборам user study", "",
         "SAM-1 vit_b, узел `L2.norm2`, все 7 токенов × 256. Вектор набора — среднее по всем его картинкам (50) "
         "разности состояний внутри картинки. Косинус — по всему тензору 7 × 256. COCO_A25 — опубликованный вектор "
         "(25 картинок половины A).", "",
         "## Косинусы векторов 13 лучших − 12 худших", "",
         "На диагонали (курсивом) — шум: косинус векторов двух непересекающихся случайных половин одного набора "
         f"(25 vs 25 картинок, среднее по {args.splits} разбиениям). Если косинус двух наборов близок к их "
         "диагоналям — направление одно и то же; если намного ниже — разное.", "",
         md_matrix(rows["halves"], h.split_half), "",
         "## Шум оценки вектора", "",
         "| набор | картинок | cos половин (25 vs 25) | cos 10 vs 10 картинок | надёжность полного вектора ≈ 2r/(1+r) | ‖v‖ |",
         "|---|---|---|---|---|---|"]
    for ds, r in h.iterrows():
        L.append(f"| {ds} | {r.n_images} | {r.split_half:+.2f} | {r.split_10:+.2f} | {r.reliability_full:.2f} | "
                 f"{r.norm:.3f} |")
    L += ["", "## Косинусы с поправкой на шум (приближённо)", "",
          "cos / √(r_i · r_j), r — надёжность полного вектора набора. Аналогия с поправкой корреляции на "
          "надёжность; для косинусов это приближение, значения > 1 — признак того, что шум оценён грубо.", "",
          md_matrix(corrected["halves"]), "",
          "## Покомпонентно: косинус с вектором COCO (50 картинок) по токенам", "",
          "| токен | " + " | ".join(tok.columns) + " |", "|---|" + "---|" * len(tok.columns)]
    for t, r in tok.iterrows():
        L.append(f"| {t} | " + " | ".join(f"{v:+.2f}" for v in r) + " |")
    for r in [x for x in RULES if x != "halves"]:
        hr = N[N.rule == r].set_index("dataset")
        L += ["", f"## Косинусы векторов {RULES[r]}", "", md_matrix(rows[r], hr.split_half)]
    rp, gr = checks.get("repro", {}), checks.get("groups", {})
    L += ["", "## Проверки", "",
          f"- половина A COCO = `--vector`: max |Δ| {checks.get('coco_a25_vs_vector', {}).get('max_abs', '-')};",
          f"- undef vs reference csv: max |Δ| {rp.get('max_abs_diff', float('nan')):.2e}, "
          f"доля > {args.repro_tol:g}: {rp.get('frac_over', float('nan')):.2%};",
          f"- группы 12/13: отклонений {gr.get('n_images_not_12_13', '-')}."]
    return "\n".join(L) + "\n"


def run(args) -> int:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    datasets = list(SUS.DATASETS)
    index = SUS.index_study(root, args.use, datasets)
    datasets = [d for d in datasets if index[d]]
    A, _ = SUS.split_halves(index["COCO"], args.split_seed)
    device = torch.device(args.device)
    predictor = SUS.load_predictor(args.checkpoint_path, args.model_type, device)
    sites = build_sites(len(predictor.model.mask_decoder.transformer.layers))
    site = next(s for s in sites if s.id == args.site)
    runner = Runner(predictor, sites, args.batch, field_grid=64, field_margin=0.25)

    per, dfs, tight_iou = {}, [], {}
    for ds in datasets:
        stems = list(index[ds])[:args.limit_images] if args.limit_images > 0 else list(index[ds])
        print(f"[{ds}] {len(stems)} images", flush=True)
        for i, stem in enumerate(stems):
            agg, df, ti = image_state(runner, site, root, stem, index[ds][stem], device)
            per[stem] = (ds, agg)
            dfs.append(df.assign(dataset=ds))
            tight_iou[stem] = ti
            if (i + 1) % 10 == 0 or i + 1 == len(stems):
                print(f"[{ds}] [{i + 1}/{len(stems)}] ({time.time() - t0:.0f}s)", flush=True)
    D = pd.concat(dfs, ignore_index=True)
    D.to_csv(out / "boxes.csv", index=False)
    np.savez_compressed(out / "per_image.npz", stems=np.asarray(list(per)),
                        datasets=np.asarray([per[s][0] for s in per]),
                        **{k: np.stack([per[s][1][k] for s in per]).astype(np.float32)
                           for k in ("top", "bot", "all", "tight", "worst")},
                        tight_iou=np.asarray([tight_iou[s] for s in per], np.float32))

    d_img = {r: OrderedDict() for r in RULES}
    for ds in datasets:
        st = [s for s in per if per[s][0] == ds]
        dd = [per_image_d(per[s][1]) for s in st]
        for r in RULES:
            d_img[r][ds] = np.stack([x[r] for x in dd])
    A_have = [s for s in A if s in per]
    for r in RULES:
        d_img[r]["COCO_A25"] = np.stack([per_image_d(per[s][1])[r] for s in A_have])
    names = [*datasets, "COCO_A25"]

    checks = {}
    if args.vector and Path(args.vector).is_file():
        ref_v = np.load(args.vector)["v"]
        diff = float(np.abs(d_img["halves"]["COCO_A25"].mean(0) - ref_v).max())
        checks["coco_a25_vs_vector"] = {"max_abs": diff, "ok": diff < 1e-6 or args.limit_images > 0}
    ref = SUS.read_reference(args.reference_csv, args.use)
    checks["repro"] = SUS.check_repro(D, ref, args)
    checks["groups"] = SUS.check_groups(D, ref, args, args.limit_images <= 0)
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=SUS._json_default), encoding="utf-8")
    failed = [k for k, r in checks.items() if isinstance(r, dict) and r.get("ok") is False]
    if failed:
        print(f"CHECK FAILED: {', '.join(failed)} -- see {out / 'checks.json'}", file=sys.stderr)
        return 1

    V, rows, corrected, N, tok = analyse(d_img, names, args)
    np.savez_compressed(out / "vectors_full.npz", **{f"{r}/{ds}": v for r, by in V.items() for ds, v in by.items()})
    for r in RULES:
        rows[r].to_csv(out / f"cos_{r}.csv")
        corrected[r].to_csv(out / f"cos_{r}_corrected.csv")
    N.to_csv(out / "noise.csv", index=False)
    tok.to_csv(out / "cos_halves_per_token_vs_COCO.csv")
    md = summary(rows, corrected, N, tok, args, checks)
    (out / "summary.md").write_text(md, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print("\n" + md.encode(enc, "replace").decode(enc))
    (out / "meta.json").write_text(json.dumps({"args": vars(args), "images": len(per),
                                               "minutes": round((time.time() - t0) / 60, 1),
                                               "created": time.strftime("%Y-%m-%d %H:%M")}, indent=2,
                                              default=SUS._json_default), encoding="utf-8")
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="user study FOR_TEST")
    p.add_argument("--use", default="m")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--site", default="L2.norm2")
    p.add_argument("--vector", default="exp_res/steer_user_study/vector.npz",
                   help="the reported COCO A vector: the half-A part must reproduce it")
    p.add_argument("--reference_csv", default="")
    p.add_argument("--splits", type=int, default=200, help="random splits for the noise level")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repro_tol", type=float, default=1e-3)
    p.add_argument("--repro_max_frac", type=float, default=0.01)
    p.add_argument("--group_tol", type=float, default=1e-3)
    p.add_argument("--limit_images", type=int, default=0, help="per dataset, for a smoke run")
    p.add_argument("--out_dir", default="exp_res/steer_full_vectors")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
