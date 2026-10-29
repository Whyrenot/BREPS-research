"""
build_paper_table.py
====================
One CSV for analysis: a row per (model, user annotation) with every method's
result and the features we have, from finished runs -- nothing is recomputed
on a model. CPU only.

SOURCES (per model; <tag> = _mp for the mobile + desktop annotations, '' for mobile only)
  exp_res/pool_<model><tag>/       build_candidate_pool.py: per_box.csv.gz (every method's IoU,
                                   the ascents' checkpoints, steering at every alpha), rows.csv.gz
                                   (the alpha regressors' features), pool.csv.gz (the undef row's
                                   f_* / x_* features, the methods' box moves, the pool's oracle)
  exp_res/cb_<model><tag>/         train_pool_catboost.py: CatBoost's out-of-fold pick, alpha*,
                                   the image folds
  exp_res/cb_<src>_on_<model><tag>/  the --transfer_from model's selectors on this pool
  exp_res/cb_<model>/, pool_<model>/ (with --mobile_tag '', when the tag is not '') the mobile-only
                                   run: its CatBoost's pick on the m rows, and a check that the m rows
                                   of this pool reproduce the mobile pool

OUTPUT  <out>.csv and <out>_columns.md (what every column is, how the groups and
  folds were made, the checks). Rows: 3 models x every user annotation of the
  ten subsets, COCO A included (in_selector_eval = False there: the steering
  vector was estimated on those images, so CatBoost neither trains nor scores
  on them).

    python scripts/build_paper_table.py --tag _mp --transfer_from SAM3 --out exp_res/paper_table_mp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["image_name", "kind", "user", "attempt"]
MEDICAL = ("ACDC", "BUID", "MedScribble")
MODES = (0, 1)
PRE = {0: "tok0", 1: "multi"}
SLOTS = ("hsel", "grad_mm0", "grad_mm1", "bon_mm0", "bon_mm1")
KIND_NAME = {"m": "mobile", "p": "desktop"}
STR = {k: str for k in KEY + ["half", "dataset", "group", "cand_kind", "method_slot", "cb_kind", "oracle_kind"]}


def _a(a: float) -> str:
    return f"a{a:g}"


def case_of(df: pd.DataFrame) -> pd.Series:
    return df.image_name + "|" + df.kind + "|" + df.user + "|" + df.attempt


def read(path: Path, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={k: v for k, v in STR.items()}, **kw)
    for k in KEY:
        if k in df.columns:
            df[k] = df[k].fillna("")
    return df


def pool_extract(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From pool.csv.gz, in chunks: the undef row's f_* / x_* (no GT), the box
    move of the methods' final candidates, and the pool's best candidate."""
    keep, best = [], []
    for ch in pd.read_csv(path, dtype=STR, chunksize=400_000):
        for k in KEY:
            ch[k] = ch[k].fillna("")
        ch["method_slot"] = ch["method_slot"].fillna("")
        ch["case"] = case_of(ch)
        sel = (ch.cand_kind == "undef") | ch.method_slot.isin(SLOTS)
        keep.append(ch.loc[sel, ["case", "cand_kind", "method_slot"] + [c for c in ch.columns if c.startswith(("f_", "x_"))]])
        i = ch.groupby("case", sort=False).target_iou.idxmax()
        best.append(ch.loc[i, ["case", "target_iou", "cand_kind", "cand_id", "step", "f_steer_alpha"]])
    K = pd.concat(keep, ignore_index=True)
    O = pd.concat(best, ignore_index=True).sort_values(["case", "target_iou"], ascending=[True, False], kind="mergesort")
    O = O.drop_duplicates("case").set_index("case")
    U = K[K.cand_kind == "undef"].drop_duplicates("case").set_index("case")
    # constant for the undef row (its own mask, the user's box unmoved) or never measured: dropped
    feats = [c for c in U.columns if c.startswith(("f_", "x_")) and U[c].nunique(dropna=True) > 1
             and c not in ("f_step", "f_step_frac", "f_is_start0", "f_steer_alpha")]
    out = U[feats].add_prefix("undef_")
    for slot in ("grad_mm0", "grad_mm1", "bon_mm0", "bon_mm1"):
        S = K[K.method_slot == slot].drop_duplicates("case").set_index("case")
        cols = [c for c in S.columns if c.startswith("f_box_") and S[c].notna().any()]
        out = out.join(S[cols].add_prefix(f"{slot}_"), how="outer")
        out[f"{slot}_head"] = S["f_head"]
    out["hsel_head"] = K[K.method_slot == "hsel"].drop_duplicates("case").set_index("case")["f_head"]
    out["pool_oracle_kind"] = O.cand_kind
    out["pool_oracle_cand_id"] = O.cand_id.astype("Int64")
    out["pool_oracle_step"] = O.step
    out["pool_oracle_alpha"] = O.f_steer_alpha
    return out, O


def image_groups(df: pd.DataFrame, by: list) -> pd.Series:
    """selector_data.image_groups(lo=0, hi=0.5) inside every `by` group: the
    round(n / 2) worst by undef_iou (ties: case order) are bottom50."""
    out = pd.Series(index=df.index, dtype=object)
    for _, s in df.sort_values("case").groupby(by, sort=False):
        o = s.undef_iou.sort_values(kind="mergesort").index
        b = int(round(0.5 * len(o)))
        out[o[:b]] = "bottom50"
        out[o[b:]] = "top50"
    return out


def one_model(model: str, a, checks: dict) -> pd.DataFrame | None:
    exp, tag = Path(a.exp), a.tag
    pool, cb = exp / f"pool_{model}{tag}", exp / f"cb_{model}{tag}"
    if not (pool / "per_box.csv.gz").is_file():
        print(f"[skip] {model}: no {pool}/per_box.csv.gz", flush=True)
        return None
    t0 = time.time()
    c = checks.setdefault(model, {})
    pm = json.loads((pool / "meta.json").read_text(encoding="utf-8"))
    alphas = pm["alphas"]
    PB = read(pool / "per_box.csv.gz")
    PB["case"] = case_of(PB)
    if PB.case.duplicated().any():
        raise SystemExit(f"{model}: duplicate boxes in {pool}/per_box.csv.gz")
    D = PB.set_index("case")
    D.insert(0, "model", model)
    D["subset"] = np.where(D.dataset == "COCO", "COCO_" + D.half.astype(str), D.dataset)
    D["domain"] = np.where(D.dataset.isin(MEDICAL), "medical", "natural")
    D["annotation"] = D.kind.map(KIND_NAME).fillna(D.kind)
    D = D.rename(columns={"fold": "steer_fold", "oracle_iou": "pool_oracle_iou"})
    # the alpha regressors' features (no GT: the IoU columns of rows.csv.gz are not read)
    R = read(pool / "rows.csv.gz")
    R["case"] = case_of(R)
    R = R.drop_duplicates("case").set_index("case").reindex(D.index)
    D["box_frac"], D["log_aspect"] = R.box_frac, R.log_aspect
    for mm, name in ((0, "undef"), (1, "hsel")):
        for f in ("stab0", "fill0", "mask_frac0"):
            D[f"{name}_{f[:-1]}"] = R[f"{PRE[mm]}_{f}"]
        for al in alphas:
            for f in ("pred", "chg", "stab", "fill"):
                D[f"steer_mm{mm}_{_a(al)}_{f}"] = R[f"{PRE[mm]}_{f}_{_a(al)}"]
            if mm == 1:
                D[f"steer_mm1_{_a(al)}_head"] = R[f"multi_head_{_a(al)}"]
    # the pool's undef features, the methods' box moves and heads, the pool's best candidate
    if (pool / "pool.csv.gz").is_file() and not a.no_pool:
        X, O = pool_extract(pool / "pool.csv.gz")
        D = D.join(X, how="left")
        c["pool_vs_per_box_oracle_max_abs"] = float((O.target_iou.reindex(D.index) - D.pool_oracle_iou).abs().max())
        c["pool_boxes_without_undef_row"] = int(D["undef_f_pred"].isna().sum()) if "undef_f_pred" in D else None
    # CatBoost of this model, out of fold
    if (cb / "per_box.csv.gz").is_file():
        C = read(cb / "per_box.csv.gz")
        C["case"] = case_of(C)
        C = C.set_index("case")
        cm = json.loads((cb / "meta.json").read_text(encoding="utf-8"))
        for mm, st in cm.get("alpha_star", {}).items():
            D[f"steer_mm{mm}_astar_alpha"] = st["alpha"]
            D[f"steer_mm{mm}_astar_iou"] = D[f"steer_mm{mm}_{_a(st['alpha'])}_iou"]
        D["in_selector_eval"] = D.index.isin(C.index)
        D["argmax_pred_iou"] = C.argmax_pred
        for col, new in (("catboost", "cb_iou"), ("cb_kind", "cb_kind"), ("cb_cand_id", "cb_cand_id"),
                         ("cb_alpha", "cb_alpha"), ("cb_step", "cb_step"), ("cb_score", "cb_score"),
                         ("cb_margin", "cb_margin")):
            if col in C.columns:
                D[new] = C[col]
        diff = (C.undef - D.undef_iou.reindex(C.index)).abs()
        c["cb_vs_pool_undef_max_abs"] = float(diff.max())
        folds = json.loads((cb / "folds.json").read_text(encoding="utf-8"))["folds"]
        f_of = {im: k for k, ims in enumerate(folds) for im in ims}
        D["cv_fold"] = D.image_name.map(f_of).fillna(-1).astype(int)
    else:
        print(f"[warn] {model}: no CatBoost run in {cb}", flush=True)
        D["in_selector_eval"] = D.subset != "COCO_A"
    # the transferred selector
    src = a.transfer_from
    tr = exp / f"cb_{src}_on_{model}{tag}"
    if src and src != model and (tr / "per_box.csv.gz").is_file():
        C = read(tr / "per_box.csv.gz")
        C["case"] = case_of(C)
        C = C.set_index("case")
        D["cbT_source"] = src
        for col, new in (("catboost", "cbT_iou"), ("cb_kind", "cbT_kind"), ("cb_alpha", "cbT_alpha"),
                         ("cb_step", "cbT_step"), ("cb_margin", "cbT_margin")):
            if col in C.columns:
                D[new] = C[col]
    # the mobile-only run: its CatBoost on the m rows, and the m rows reproduced
    if a.mobile_tag is not None and a.mobile_tag != tag:
        mp_pool, mp_cb = exp / f"pool_{model}{a.mobile_tag}", exp / f"cb_{model}{a.mobile_tag}"
        if (mp_cb / "per_box.csv.gz").is_file():
            C = read(mp_cb / "per_box.csv.gz")
            C["case"] = case_of(C)
            C = C.set_index("case")
            D["cb_mobile_trained_iou"] = C.catboost
            D["cb_mobile_trained_kind"] = C.cb_kind
        if (mp_pool / "per_box.csv.gz").is_file():
            M = read(mp_pool / "per_box.csv.gz")
            M["case"] = case_of(M)
            M = M.set_index("case")
            common = M.index.intersection(D.index)
            rep = {}
            for col in ["undef_iou", "hsel_iou", "grad_mm0_iou", "grad_mm1_iou", "bon_mm0_iou", "bon_mm1_iou"] + \
                       [f"steer_mm{mm}_{_a(al)}_iou" for mm in MODES for al in alphas[::3]]:
                if col in M.columns:
                    d = (M.loc[common, col] - D.loc[common, col]).abs()
                    rep[col] = {"max_abs": float(d.max()), "share_over_1e-3": float((d > 1e-3).mean())}
            c["mobile_rows_vs_mobile_pool"] = {"mobile_boxes": int(len(M)), "found_here": int(len(common)), "diff": rep}
    # groups: as the runs (inside the image over all its annotations) and inside the image and annotation type
    D = D.reset_index()
    D["group_kind"] = image_groups(D, ["image_name", "kind"])
    D["undef_rank_img"] = D.groupby("image_name").undef_iou.rank(pct=True)
    D["n_ann_img"] = D.groupby("image_name").case.transform("size")
    D["n_ann_img_kind"] = D.groupby(["image_name", "kind"]).case.transform("size")
    c["boxes"] = int(len(D))
    c["by_kind"] = D.kind.value_counts().to_dict()
    c["in_selector_eval"] = int(D.in_selector_eval.sum())
    print(f"{model}: {len(D)} boxes ({c['by_kind']}), {int(D.in_selector_eval.sum())} scored by CatBoost, "
          f"{D.shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)
    return D


FRONT = ["model", "image_name", "dataset", "subset", "domain", "kind", "annotation", "user", "attempt", "case", "half",
         "steer_fold", "cv_fold", "in_selector_eval", "group", "group_kind", "undef_rank_img", "n_ann_img",
         "n_ann_img_kind", "clean_iou", "pool_oracle_iou", "n_candidates"]
MAIN = ["undef_iou", "hsel_iou", "grad_mm0_iou", "grad_mm1_iou", "bon_mm0_iou", "bon_mm1_iou", "steer_mm0_astar_iou",
        "steer_mm1_astar_iou", "argmax_pred_iou", "cb_iou", "cbT_iou", "cb_mobile_trained_iou", "pool_oracle_iou"]


def columns_md(df: pd.DataFrame, a, checks: dict, overview: pd.DataFrame) -> str:
    al = sorted({c.split("_")[2] for c in df.columns if c.startswith("steer_mm0_a") and c.endswith("_iou")
                 and "astar" not in c}, key=lambda s: float(s[1:]))
    L = [f"# {Path(a.out).name}.csv: что в каждом столбце", "",
         f"Одна строка — одна модель × одна пользовательская разметка. Модели: {', '.join(df.model.unique())}; "
         f"строк {len(df)}; столбцов {df.shape[1]}. Разметки: m — мобильная, p — с ПК (`annotation`). Собрано "
         f"{time.strftime('%Y-%m-%d %H:%M')} из `{a.exp}/pool_<model>{a.tag}`, `cb_<model>{a.tag}`"
         + (f", `cb_{a.transfer_from}_on_<model>{a.tag}`" if a.transfer_from else "") + ".", "",
         "## Как устроены данные", "",
         "- **Бокс** — tight bbox маски пользователя; **GT** — маска набора. Все `*_iou` — IoU маски с GT на полном "
         "разрешении. `*_pred` — предсказанный моделью IoU (голова IoU), GT не нужен.",
         "- **Режимы выхода:** `mm0` — multimask=False, как модель поставляется (у SAM2.1 / SAM3 с fallback по "
         "стабильности); `mm1` — multimask=True, голова с максимальным предсказанным IoU (head-select).",
         "- **Методы:** undef — один выход на боксе пользователя; hsel — head-select на нём же; grad — 30 шагов "
         "подъёма предсказанного IoU по координатам бокса (Adam, lr 1); BoN — 16 боксов с шумом размера и центра, "
         "берётся максимум предсказанного IoU; steering — вектор (разность средних лучших и худших разметок COCO A, "
         "мобильная разметка) прибавляется к токенам узла `L2.norm3` с силой α; CatBoost — выбор одного из ~66 "
         "кандидатов пула (YetiRank, вне фолда).",
         "- **COCO A** (`subset == COCO_A`, 25 картинок): на них считался вектор steering (там он кросс-фит по 5 "
         "фолдам, `steer_fold`). CatBoost их не видит: `in_selector_eval = False`, столбцы `cb_*` пустые. α* "
         "steering выбрана на них по среднему IoU.",
         "- **Фолды CatBoost** (`cv_fold`): 5 фолдов по картинкам, стратификация по набору; `cb_*` у картинки — от "
         "модели, обученной на других фолдах. `cbT_*` — селектор другой модели (`cbT_source`), модель того же фолда.",
         "- **Группы:** `group` — внутри картинки по `undef_iou` среди всех её разметок: худшие round(n/2) — "
         "bottom50, остальные — top50 (так считались все таблицы прогонов); `group_kind` — то же внутри картинки и "
         "типа разметки (как в мобильных таблицах, 12 / 13).",
         "- Признаки без GT пригодны как входы селектора; столбцы `*_iou` и `pool_oracle_*` — только для оценки.", "",
         "## Столбцы", "", "| столбец | что это |", "|---|---|"]
    desc = [
        ("model", "SAM (vit_b), SAM2.1 (hiera_base_plus), SAM3"),
        ("image_name / dataset / subset", "картинка; набор; набор с половиной COCO (COCO_A / COCO_B)"),
        ("domain", "natural (7 наборов) / medical (ACDC, BUID, MedScribble)"),
        ("kind / annotation / user / attempt", "тип разметки m / p (mobile / desktop), пользователь, попытка"),
        ("case", "ключ разметки: image_name, kind, user, attempt через «\\|» (одинаков у всех моделей)"),
        ("half, steer_fold, cv_fold, in_selector_eval", "половина COCO; фолд steering на COCO A; фолд CatBoost; "
                                                         "оценивалась ли разметка CatBoost"),
        ("group, group_kind, undef_rank_img, n_ann_img, n_ann_img_kind",
         "группы (см. выше); доля разметок картинки с undef_iou не выше; число разметок картинки (всего / этого типа)"),
        ("clean_iou", "IoU одного выхода на tight bbox GT (идеальный бокс)"),
        ("box_frac, log_aspect", "площадь бокса пользователя / площадь кадра; log(ширина / высота)"),
        ("undef_iou / _pred / _head; hsel_iou / _pred / _head", "один выход и head-select: IoU, предсказанный IoU, "
                                                                 "номер головы (0 — mask token 0, 1–3 — multimask)"),
        ("undef_stab, undef_fill, undef_mask_frac; hsel_stab, …", "без GT: stability логитов, маска / бокс, маска / кадр "
                                                           "(признаки регрессора α при α = 0)"),
        ("undef_f_*", "признаки маски undef из пула (форма маски, компоненты, положение бокса в кадре, …)"),
        ("undef_x_iou_*", "IoU маски undef с маской метода (hsel, grad, BoN): насколько метод меняет маску, без GT"),
        ("grad_mm{0,1}_iou / _pred / _head", "grad, 30 шагов, без / с multimask"),
        ("grad_mm{0,1}_s{k}_iou", "IoU после k шагов подъёма (k = 0, 5, …, 30)"),
        ("grad_mm{0,1}_f_box_*, bon_mm{0,1}_f_box_*", "насколько метод сдвинул бокс относительно бокса пользователя"),
        ("bon_mm{0,1}_iou / _pred / _head", "BoN без / с multimask; `_oracle_iou` — лучший из 16 боксов по GT"),
        (f"steer_mm{{0,1}}_a{{α}}_iou / _pred / _chg / _stab / _fill",
         f"steering при α ∈ {{{', '.join(x[1:] for x in al)}}}: IoU, предсказанный IoU, изменение маски к α = 0, "
         "stability, маска / бокс; у mm1 ещё `_head`"),
        ("steer_mm{0,1}_astar_alpha / _astar_iou", "α*, выбранная на COCO A, и IoU при ней"),
        ("argmax_pred_iou", "кандидат пула с максимальным предсказанным IoU (без обучения)"),
        ("cb_iou, cb_kind, cb_cand_id, cb_alpha, cb_step, cb_score, cb_margin",
         "CatBoost этой модели вне фолда: IoU выбранного, его семейство (undef, hsel, grad_mm0, …, steer_mm1), номер "
         "в семействе, α (steering), шаг (grad), score и отрыв от второго кандидата"),
        ("cbT_*", "то же для селектора, перенесённого с `cbT_source`"),
        ("cb_mobile_trained_iou / _kind", "CatBoost, обученный только на мобильной разметке (только строки m)"),
        ("pool_oracle_iou, pool_oracle_kind / _cand_id / _step / _alpha", "лучший кандидат пула по GT (потолок)"),
        ("n_candidates", "число кандидатов в пуле разметки"),
    ]
    L += [f"| `{n}` | {d} |" for n, d in desc]
    try:
        ov = overview.to_markdown(index=False)
    except ImportError:                          # to_markdown needs tabulate
        ov = "```\n" + overview.to_string(index=False) + "\n```"
    L += ["", "## Средние IoU (только строки, которые оценивал CatBoost)", "", ov, "", "## Проверки", "",
          "```", json.dumps(checks, indent=1, ensure_ascii=False, default=str), "```", ""]
    return "\n".join(L)


def overview_table(df: pd.DataFrame) -> pd.DataFrame:
    E = df[df.in_selector_eval]
    cols = [c for c in MAIN if c in E.columns]
    O = E.groupby(["model", "annotation", "domain"])[cols].mean().round(4).reset_index()
    A = E.groupby(["model", "annotation"])[cols].mean().round(4).reset_index().assign(domain="all")
    return pd.concat([O, A], ignore_index=True).sort_values(["model", "annotation", "domain"]).reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exp", default="exp_res")
    p.add_argument("--models", nargs="*", default=["SAM", "SAM2.1", "SAM3"])
    p.add_argument("--tag", default="_mp", help="annotation-set suffix of the run dirs ('' = mobile only)")
    p.add_argument("--transfer_from", default="SAM3", help="model whose selectors were applied to the others ('' = none)")
    p.add_argument("--mobile_tag", default="", help="suffix of the mobile-only runs to join and check against "
                                                     "(none: --mobile_tag none)")
    p.add_argument("--no_pool", action="store_true", help="skip pool.csv.gz (the undef features, box moves, oracle)")
    p.add_argument("--out", default="exp_res/paper_table_mp")
    a = p.parse_args(argv)
    if a.mobile_tag == "none":
        a.mobile_tag = None
    t0 = time.time()
    checks: dict = {}
    parts = [d for d in (one_model(m, a, checks) for m in a.models) if d is not None]
    if not parts:
        raise SystemExit("no model has a pool under --exp with this --tag")
    df = pd.concat(parts, ignore_index=True, sort=False)
    front = [c for c in FRONT if c in df.columns]
    main_cols = [c for c in MAIN if c in df.columns and c not in front]
    rest = sorted(c for c in df.columns if c not in front and c not in main_cols)
    df = df[front + main_cols + rest]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.6g")
    ov = overview_table(df)
    out.with_name(out.name + "_columns.md").write_text(columns_md(df, a, checks, ov), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ov.to_string(index=False))
    for m, c in checks.items():
        r = c.get("mobile_rows_vs_mobile_pool")
        if r:
            worst = max(r["diff"].items(), key=lambda kv: kv[1]["max_abs"]) if r["diff"] else ("-", {"max_abs": np.nan})
            print(f"{m}: m rows vs the mobile pool: {r['found_here']}/{r['mobile_boxes']} boxes found, worst "
                  f"{worst[0]} max |d| {worst[1]['max_abs']:.2g}", flush=True)
    print(f"wrote {out.with_suffix('.csv')} ({df.shape[0]} rows x {df.shape[1]} columns, "
          f"{out.with_suffix('.csv').stat().st_size / 1e6:.0f} MB) and {out.name}_columns.md "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
