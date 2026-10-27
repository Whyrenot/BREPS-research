"""
steer_userstudy_report.py
=========================
The user-study steering results as a report: one figure and two tables, read
from finished runs (no GPU, no refit of CatBoost).

  figure    IoU of bottom50 / top50 against the number of annotations per image
            the COCO A vector was estimated on (1+1, 3+3, 5+5 extreme boxes, all
            13+12), steering by steer_gate's GBM gain regressor; the undefended
            IoU of each group as a flat line; natural and medical test sets in
            two panels (the mean over the sets, each weighted equally, with the
            95% image-bootstrap band of steer_variants)          <- steer_variants
  table 2   per dataset: undefended, COCO vectors of as many images as the
            dataset's own (10; the mean over the COCO draws), the own vector,
            bottom50 / top50, % to undefended; the own vector's gain over the
            COCO ones in % of them with a paired 95% image-bootstrap CI, its
            range over the single draws, and the cosine          <- steer_variants
  table 4   the vectors with the tight GT box as a positive against the
            full-halves COCO A vector, every test set              <- steer_variants
  table 5   one vector for all: the pool of every dataset's own draw, and the
            pool without the dataset at hand, against its own vector <- steer_variants
  table 6   COCO half B: the tight-GT-box-minus-worst-box vector against the
            full-halves one, under every way of choosing alpha     <- steer_variants
  A part whose inputs a run lacks is skipped (and listed at the end).
  table 3   per dataset: the CatBoost selector without and with the steering
            candidates, bottom50 / top50, % to undefended; the gain in % of the
            CatBoost without steering, paired CI                  <- selector_steer

Table 2 refits only what steer_variants fitted (alpha regressors, fixed
seeds) and checks that its IoUs equal table_exp3.csv; table 3 reads the
out-of-fold picks of selector_steer's per_case.csv and checks them against
its table.csv.

    python scripts/steer_userstudy_report.py --variants_dir exp_res/steer_variants \\
        --selector_dir results/selector_steer --out_dir exp_res/steer_userstudy_report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import steer_gate as G  # noqa: E402
import steer_variants as SV  # noqa: E402

KEY = SV.KEY
MEDICAL = ("ACDC", "BUID", "MedScribble")
NATURAL_ORDER = ["COCO", "COCO_B", "Berkeley", "GrabCut", "DAVIS", "PASCAL", "ADE20K", "TETRIS"]
GROUPS = ("bottom50", "top50")
NAME = {"COCO_B": "COCO (половина B)", "COCO": "COCO"}

# figure: the reference palette's first two categorical slots (validated, light surface)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = {"bottom50": "#2a78d6", "top50": "#eb6834"}
K_POINTS = [("ext1", 2, "1 + 1"), ("ext3", 6, "3 + 3"), ("ext5", 10, "5 + 5"), ("all", 25, "13 + 12\n(все 25)")]


def ordered(blocks):
    nat = [b for b in NATURAL_ORDER if b in blocks]
    med = [b for b in MEDICAL if b in blocks]
    return nat + med + [b for b in blocks if b not in nat and b not in med]


# ---------------------------------------------------------------------------
# bootstrap over images
# ---------------------------------------------------------------------------

def image_sums(x: np.ndarray, img: np.ndarray, codes: np.ndarray):
    inv = np.searchsorted(codes, img)
    return np.bincount(inv, weights=x, minlength=len(codes)), np.bincount(inv, minlength=len(codes)).astype(float)


def boot_means(cols: dict, img: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap means of every column, the same image resamples for all
    (paired), the boxes of an image drawn together."""
    codes = np.unique(img)
    idx = np.random.default_rng(seed).integers(0, len(codes), size=(n_boot, len(codes)))
    out = {}
    for k, x in cols.items():
        S, N = image_sums(x, img, codes)
        out[k] = S[idx].sum(1) / N[idx].sum(1)
    return out


def rel_ci(new: np.ndarray, base: np.ndarray):
    r = (new / base - 1) * 100
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def figure_data(T12: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    blocks = {"natural": [b for b in T12.block.unique() if b.startswith("среднее, натур")],
              "medical": [b for b in T12.block.unique() if b.startswith("среднее, медиц")]}
    label = {"ext1": "1 лучших / 1 худших", "ext3": "3 лучших / 3 худших", "ext5": "5 лучших / 5 худших",
             "all": "все разметки (13 лучших / 12 худших)"}
    for panel, bl in blocks.items():
        if not bl:
            continue
        B = T12[T12.block == bl[0]]
        for var, x, tick in K_POINTS:
            for g in GROUPS:
                r = B[(B.method == f"{label[var]}: {method}") & (B.group == g)].iloc[0]
                rows.append({"panel": panel, "block": bl[0], "variant": var, "x": x, "tick": tick, "group": g,
                             "iou": r.iou, "undef": r.base, "lo": r.base + r.d_lo, "hi": r.base + r.d_hi,
                             "pct": r.pct})
    return pd.DataFrame(rows)


def plot_figure(F: pd.DataFrame, path: Path, method_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    titles = {"natural": "Натуральные: COCO (половина B), Berkeley, GrabCut",
              "medical": "Медицина: ACDC, BUID, MedScribble"}
    panels = [p for p in ("natural", "medical") if p in set(F.panel)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.9), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    xs = [x for _, x, _ in K_POINTS]
    for ax, panel in zip(axes[0], panels):
        P = F[F.panel == panel]
        ax.set_facecolor(SURFACE)
        for g in GROUPS:
            s = P[P.group == g].sort_values("x")
            c = SERIES[g]
            u = float(s.undef.iloc[0])
            ax.fill_between(s.x, s.lo, s.hi, color=c, alpha=0.10, lw=0)
            ax.axhline(u, color=c, lw=1.5, ls=(0, (5, 3)), alpha=0.9)
            ax.plot(s.x, s.iou, color=c, lw=2, solid_capstyle="round", solid_joinstyle="round", marker="o", ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            # every label on the side of its line that the other line of the group is not on
            up = float(s.iou.mean()) >= u                       # steering above its undef line
            ax.annotate(f"undef {g}  {u:.3f}", (0.4, u), xytext=(0, -4 if up else 4), textcoords="offset points",
                        va="top" if up else "bottom", ha="left", fontsize=8.5, color=INK2)
            for _, r in s[s.x.isin([xs[0], xs[-1]])].iterrows():
                above = r.iou >= u
                ax.annotate(f"{r.iou:.3f} ({r.pct:+.1f}%)", (r.x, r.iou), xytext=(0, 9 if above else -9),
                            textcoords="offset points", ha="center", va="bottom" if above else "top",
                            fontsize=8.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([t for _, _, t in K_POINTS])
        ax.set_xlim(0, 27)
        lo = min(F[F.panel == panel][["lo", "undef"]].min())
        hi = max(F[F.panel == panel][["hi", "undef", "iou"]].max())
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad, hi + pad * 1.6)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(titles[panel], loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("разметок на картинку для вектора: лучших + худших (25 картинок COCO A)")
        ax.set_ylabel("IoU")
    handles = [Line2D([], [], color=SERIES["bottom50"], lw=2, marker="o", ms=7, markeredgecolor=SURFACE,
                      label="steering, bottom50"),
               Line2D([], [], color=SERIES["top50"], lw=2, marker="o", ms=7, markeredgecolor=SURFACE,
                      label="steering, top50"),
               Line2D([], [], color=SERIES["bottom50"], lw=1.5, ls=(0, (5, 3)), label="undef, bottom50"),
               Line2D([], [], color=SERIES["top50"], lw=1.5, ls=(0, (5, 3)), label="undef, top50"),
               Patch(color=MUTED, alpha=0.25, lw=0, label="95% CI (бутстреп по картинкам)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Steering SAM-1: IoU от числа разметок на картинку, по которым считается вектор ({method_label})",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables 2 and 4: one vector against another, on the same boxes
# ---------------------------------------------------------------------------

class Picks:
    """steer_variants' rows with every variant's alpha methods refitted as
    steer_variants fits them (same training rows, same seeds): the IoU a
    variant's --method picks on any of its boxes."""

    def __init__(self, vdir: Path, method: str):
        self.D = pd.read_csv(vdir / "rows.csv.gz", dtype={k: str for k in KEY})
        self.meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))["args"]
        self.alphas = list(self.meta["alphas"])
        fixed = [a for a in self.meta["fixed_alphas"] if a in self.alphas]
        self.method, self.fitted = method, {}
        for v in dict.fromkeys(self.D.variant):
            tr = self.D[(self.D.variant == v) & (self.D.role == "train")].reset_index(drop=True)
            if len(tr):
                self.fitted[v] = SV.variant_methods(tr, self.alphas, fixed, self.meta["seed"])[0]
        z = np.load(vdir / "vectors.npz")
        self.vec = {k[:-2]: z[k] for k in z.files if k.endswith("/v")}

    def on(self, table: str, variant: str, keys: pd.DataFrame | None = None, method: str | None = None) -> pd.DataFrame:
        """The variant's rows of a table (in the order of `keys` when given),
        with the IoU its method (--method unless given) picks as column `pick`."""
        Dt = self.D[(self.D.table == table) & (self.D.variant == variant)]
        if keys is not None:
            Dt = Dt.set_index(KEY).loc[pd.MultiIndex.from_frame(keys[KEY])].reset_index()
        Dt = Dt.reset_index(drop=True)
        W = G.iou_matrix(Dt, self.alphas)
        return Dt.assign(pick=W[np.arange(len(Dt)), self.fitted[variant](Dt)[method or self.method]])

    def cos(self, a: str, b: str) -> float:
        x, y = self.vec[a].ravel(), self.vec[b].ravel()
        return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))

    def norm_ratio(self, a: str, b: str) -> float:
        return float(np.linalg.norm(self.vec[a]) / np.linalg.norm(self.vec[b]))


def compare_blocks(per: "OrderedDict[str, pd.DataFrame]", cols, base: str, args, extra=None) -> pd.DataFrame:
    """per: block -> boxes with image_name, group, undef and a column per
    vector. Per block and group: every column's IoU and % to undef, and its
    gain over `base` in % of it with a paired 95% image-bootstrap CI; then the
    means over the blocks (natural / medical / all), each block weighted
    equally, bootstrapped within each block. extra(block, group, boxes or
    None, blocks) adds columns to a row."""
    rows, boots = [], {}

    def row(block, g, n, n_img, m, b):
        r = {"block": block, "group": g, "n": n, "n_images": n_img, "undef": m["undef"]}
        for c in cols:
            r[c] = m[c]
            r[f"{c}_pct"] = (m[c] / m["undef"] - 1) * 100
            if c != base:
                r[f"{c}_gain"] = (m[c] / m[base] - 1) * 100
                r[f"{c}_lo"], r[f"{c}_hi"] = rel_ci(b[c], b[base])
        return r

    for ti, (t, P) in enumerate(per.items()):
        for g in GROUPS:
            s = P[P.group == g]
            b = boot_means({c: s[c].to_numpy(float) for c in ["undef", *cols]}, s.image_name.to_numpy(),
                           args.n_boot, args.seed + ti)
            boots[(t, g)] = b
            r = row(t, g, len(s), s.image_name.nunique(), {c: s[c].mean() for c in ["undef", *cols]}, b)
            if extra is not None:
                r.update(extra(t, g, s, [t]))
            rows.append(r)
    T = pd.DataFrame(rows)
    nat = [t for t in per if t not in MEDICAL]
    med = [t for t in per if t in MEDICAL]
    for label, ts in ((f"среднее, натуральные ({len(nat)})", nat), (f"среднее, медицина ({len(med)})", med),
                      (f"среднее по {len(per)}", list(per))):
        if len(ts) < 2:
            continue
        for g in GROUPS:
            s = T[T.block.isin(ts) & (T.group == g)]
            b = {c: np.mean([boots[(t, g)][c] for t in ts], axis=0) for c in ["undef", *cols]}
            r = row(label, g, int(s.n.sum()), int(s.n_images.sum()), {c: s[c].mean() for c in ["undef", *cols]}, b)
            if extra is not None:
                r.update(extra(label, g, None, ts))
            rows.append(r)
    return pd.DataFrame(rows)


def own_vs_coco_sub(pk: Picks, vdir: Path, args) -> tuple:
    """Table 2: every dataset's own vector against COCO vectors of as many
    images (the mean over the draws: the expected COCO vector of that size),
    on the dataset's own test boxes; and against the 25-image COCO A one."""
    own_test = OrderedDict(json.loads((vdir / "own_test.json").read_text(encoding="utf-8")))
    n = pk.meta["own_train"]
    draws = sorted((v for v in pk.fitted if v.startswith(f"coco{n}_d")), key=lambda v: int(v.split("_d")[1]))
    if not draws:
        raise SystemExit(f"no coco{n}_d* variants in {vdir}: rerun steer_variants with --coco_sub_draws")
    T3 = pd.read_csv(vdir / "table_exp3.csv")
    per, check, comp = OrderedDict(), {}, {}
    for t, (ds, _) in own_test.items():
        # a COCO draw that IS the own vector (COCO: draw 0) is no comparator for it
        comp[t] = [d for d in draws if not np.array_equal(pk.vec[d], pk.vec[f"own_{ds}"])]
        own = pk.on(t, f"own_{ds}")
        own = own[own.role == "eval"].reset_index(drop=True)
        P = own[["image_name", "group", "undef_iou"]].rename(columns={"undef_iou": "undef"})
        P["own"] = own.pick
        for d in draws:
            sb = pk.on(t, d, own)
            if not (sb.undef_iou.to_numpy() == own.undef_iou.to_numpy()).all():
                raise SystemExit(f"{t}: {d} and own_{ds} rows are not the same boxes")
            P[d] = sb.pick.to_numpy()
        P["coco_sub"] = P[comp[t]].mean(1)
        P["coco25"] = pk.on(t, "all", own).pick.to_numpy()
        per[t] = P
        ref = T3[(T3.block == t) & (T3.method == f"свой вектор ({n} изобр.): {pk.method}") & (T3.group == "all")].iou
        check[t] = float(abs(P.own.mean() - ref.iloc[0])) if len(ref) else None
    cos = {t: float(np.mean([pk.cos(f"own_{ds}", d) for d in comp[t]])) for t, (ds, _) in own_test.items()}

    def extra(block, g, s, ts):
        # the own vector's gain over each COCO draw alone: how much the draw matters
        own_m = np.mean([per[t][per[t].group == g].own.mean() for t in ts])
        ds_ = [d for d in draws if all(d in comp[t] for t in ts)]    # comparators of every table of the block
        dg = [(own_m / np.mean([per[t][per[t].group == g][d].mean() for t in ts]) - 1) * 100 for d in ds_]
        return {"cos_sub": cos.get(block, np.nan), "draw_min": min(dg), "draw_max": max(dg)}

    T = compare_blocks(per, ["coco_sub", "own", "coco25"], "coco_sub", args, extra)
    if "own_COCO" in pk.vec:
        check["own_COCO_vs_draw0_max_abs"] = float(np.abs(pk.vec["own_COCO"] - pk.vec[f"coco{n}_d0"]).max())
    return T, check, len(draws), n


def md_table2(T: pd.DataFrame, n_draws: int, n: int) -> str:
    L = [f"| набор | изобр. | cos(v_свой, v_COCO-{n}) | undef b50 | undef t50 | COCO-{n} b50 | COCO-{n} t50 "
         f"| свой b50 | свой t50 | свой к COCO-{n}, b50 [95% CI] | свой к COCO-{n}, t50 [95% CI] "
         f"| свой к каждой из {n_draws} выборок COCO-{n}, b50 / t50: мин … макс |",
         "|" + "---|" * 12]
    for block in dict.fromkeys(T.block):
        s = T[T.block == block].set_index("group")
        b, t = s.loc["bottom50"], s.loc["top50"]
        name = NAME.get(block, block) + (" (медицина)" if block in MEDICAL else "") + (" ¹" if block == "COCO_B" else "")
        bold = "**" if block.startswith("среднее") else ""
        cos = "—" if not np.isfinite(b.cos_sub) else f"{b.cos_sub:+.2f}"
        L.append(f"| {bold}{name}{bold} | {b.n_images} | {cos} | {b.undef:.4f} | {t.undef:.4f} | "
                 f"{b.coco_sub:.4f} ({b.coco_sub_pct:+.2f}%) | {t.coco_sub:.4f} ({t.coco_sub_pct:+.2f}%) | "
                 f"{b.own:.4f} ({b.own_pct:+.2f}%) | {t.own:.4f} ({t.own_pct:+.2f}%) | "
                 f"{bold}{b.own_gain:+.2f}%{bold} [{b.own_lo:+.2f}, {b.own_hi:+.2f}] | "
                 f"{bold}{t.own_gain:+.2f}%{bold} [{t.own_lo:+.2f}, {t.own_hi:+.2f}] | "
                 f"{b.draw_min:+.1f} … {b.draw_max:+.1f} / {t.draw_min:+.1f} … {t.draw_max:+.1f} |")
    return "\n".join(L)


TIGHT = OrderedDict([("all", "13 лучших / 12 худших (базовый)"), ("tight", "tight bbox GT / 12 худших"),
                     ("tight_top", "13 лучших + tight bbox GT / 12 худших"),
                     ("tight_all", "tight bbox GT / все 25 разметок"),
                     ("tight_worst", "tight bbox GT / 1 худшая")])


def tight_vs_base(pk: Picks, args) -> pd.DataFrame | None:
    """Table 4: the vectors with the tight GT box as a positive against the
    full-halves one, all on COCO A, on every test table."""
    vs = [v for v in TIGHT if v in pk.fitted]
    if len(vs) < 2:
        return None
    per = OrderedDict()
    for t in ordered([t for t in dict.fromkeys(pk.D.table) if t != "COCO_A"]):
        base = pk.on(t, "all")
        P = base[["image_name", "group", "undef_iou"]].rename(columns={"undef_iou": "undef"})
        for v in vs:
            P[v] = pk.on(t, v, base).pick.to_numpy()
        per[t] = P
    T = compare_blocks(per, vs, "all", args)
    for v in vs:
        T[f"cos_{v}"] = pk.cos(v, "all")
        T[f"norm_{v}"] = pk.norm_ratio(v, "all")
    return T


def md_long(T: pd.DataFrame, labels: "OrderedDict[str, str]", base: str, base_name: str, cos_name: str) -> str:
    """One row per block x vector: IoU (% to undef) per group, the gain over
    `base` with its CI, the vector's cosine / norm against the base one."""
    vs = [v for v in labels if v in T.columns]
    L = [f"| набор | вектор | {cos_name} | ‖v‖ / ‖v_{base_name}‖ | bottom50 | top50 | к {base_name}, b50 [95% CI] "
         f"| к {base_name}, t50 [95% CI] |", "|" + "---|" * 8]
    for block in dict.fromkeys(T.block):
        s = T[T.block == block].set_index("group")
        b, t = s.loc["bottom50"], s.loc["top50"]
        name = NAME.get(block, block) + (" (медицина)" if block in MEDICAL else "")
        bold = "**" if block.startswith("среднее") else ""
        L.append(f"| {bold}{name}{bold} | undef | | | {b.undef:.4f} | {t.undef:.4f} | | |")
        for v in vs:
            if v == base:
                gb = gt = "—"
            else:
                gb = f"{bold}{b[v + '_gain']:+.2f}%{bold} [{b[v + '_lo']:+.2f}, {b[v + '_hi']:+.2f}]"
                gt = f"{bold}{t[v + '_gain']:+.2f}%{bold} [{t[v + '_lo']:+.2f}, {t[v + '_hi']:+.2f}]"
            c, nr = b.get("cos_" + v, np.nan), b.get("norm_" + v, np.nan)
            c = "—" if not np.isfinite(c) else f"{c:+.2f}"
            nr = "—" if not np.isfinite(nr) else f"{nr:.2f}"
            L.append(f"| | {labels[v]} | {c} | {nr} | {b[v]:.4f} ({b[v + '_pct']:+.2f}%) | "
                     f"{t[v]:.4f} ({t[v + '_pct']:+.2f}%) | {gb} | {gt} |")
    return "\n".join(L)


def md_table4(T: pd.DataFrame) -> str:
    return md_long(T, TIGHT, "all", "базовому", "cos с базовым")


UNIVERSAL = OrderedDict([("own", "свой (10 изобр. набора)"), ("pool", "общий: 6 наборов × 10 изобр."),
                         ("lodo", "общий без этого набора: 5 × 10 изобр."), ("coco25", "COCO A, 25 изобр.")])


def universal(pk: Picks, vdir: Path, args) -> pd.DataFrame | None:
    """Table 5: one vector for every dataset. On each dataset's own test boxes:
    its own vector, the vector pooled over every dataset's own draw, the pool
    without this dataset (an unseen domain), the COCO A one."""
    if "pool" not in pk.fitted:
        return None
    own_test = OrderedDict(json.loads((vdir / "own_test.json").read_text(encoding="utf-8")))
    per, geo = OrderedDict(), {}
    for t, (ds, _) in own_test.items():
        if f"lodo_{ds}" not in pk.fitted:
            continue
        own = pk.on(t, f"own_{ds}")
        own = own[own.role == "eval"].reset_index(drop=True)
        P = own[["image_name", "group", "undef_iou"]].rename(columns={"undef_iou": "undef"})
        P["own"] = own.pick
        for col, v in (("pool", "pool"), ("lodo", f"lodo_{ds}"), ("coco25", "all")):
            sb = pk.on(t, v, own)
            if not (sb.undef_iou.to_numpy() == own.undef_iou.to_numpy()).all():
                raise SystemExit(f"{t}: {v} and own_{ds} rows are not the same boxes")
            P[col] = sb.pick.to_numpy()
        per[t] = P
        geo[t] = {f"cos_{c}": pk.cos(v, f"own_{ds}") for c, v in (("pool", "pool"), ("lodo", f"lodo_{ds}"),
                                                                    ("coco25", "all"))}
        geo[t].update({f"norm_{c}": pk.norm_ratio(v, f"own_{ds}") for c, v in (("pool", "pool"),
                                                                                ("lodo", f"lodo_{ds}"), ("coco25", "all"))})

    def extra(block, g, s, ts):
        return geo.get(block, {})

    return compare_blocks(per, list(UNIVERSAL), "own", args, extra)


# ---------------------------------------------------------------------------
# Table 3: CatBoost with and without the steering candidates
# ---------------------------------------------------------------------------

def table3(sdir: Path, args) -> tuple:
    C = pd.read_csv(sdir / "per_case.csv")
    ref = pd.read_csv(sdir / "table.csv")
    blocks = [(ds, C.dataset == ds) for ds in ordered(list(C.dataset.unique()))]
    nat = ~C.dataset.isin(MEDICAL)
    blocks += [(f"натуральные ({C[nat].dataset.nunique()})", nat), (f"медицина ({C[~nat].dataset.nunique()})", ~nat),
               (f"все {C.dataset.nunique()} наборов", pd.Series(True, index=C.index))]
    rows, check = [], 0.0
    for bi, (name, m) in enumerate(blocks):
        B = C[np.asarray(m)]
        for g in GROUPS:
            s = B[B.group == g]
            bm = boot_means({c: s[c].to_numpy(float) for c in ("cb_t4", "cb_t4s")}, s.image_name.to_numpy(),
                            args.n_boot, args.seed + bi)
            u, a, b = s.undef.mean(), s.cb_t4.mean(), s.cb_t4s.mean()
            lo, hi = rel_ci(bm["cb_t4s"], bm["cb_t4"])
            rows.append({"block": name, "group": g, "n": len(s), "undef": u, "t4": a, "t4_pct": (a / u - 1) * 100,
                         "t4s": b, "t4s_pct": (b / u - 1) * 100, "gain_pct": (b / a - 1) * 100, "lo": lo, "hi": hi})
            r = ref[(ref.block == name) & (ref.method == "cb_t4s") & (ref.group == g)].iou
            if len(r):
                check = max(check, abs(r.iloc[0] - b))
    return pd.DataFrame(rows), check


def md_table3(T: pd.DataFrame) -> str:
    L = ["| набор | undef b50 | undef t50 | CatBoost без steering b50 | CatBoost без steering t50 "
         "| CatBoost со steering b50 | CatBoost со steering t50 | выигрыш b50 [95% CI] | выигрыш t50 [95% CI] |",
         "|" + "---|" * 9]
    for block in dict.fromkeys(T.block):
        s = T[T.block == block].set_index("group")
        b, t = s.loc["bottom50"], s.loc["top50"]
        bold = "**" if "(" in block or block.startswith("все") else ""      # the means, not a dataset
        name = block + (" (медицина)" if block in MEDICAL else "")
        L.append(f"| {bold}{name}{bold} | {b.undef:.4f} | {t.undef:.4f} | {b.t4:.4f} ({b.t4_pct:+.2f}%) | "
                 f"{t.t4:.4f} ({t.t4_pct:+.2f}%) | {b.t4s:.4f} ({b.t4s_pct:+.2f}%) | {t.t4s:.4f} ({t.t4s_pct:+.2f}%) | "
                 f"{bold}{b.gain_pct:+.2f}%{bold} [{b.lo:+.2f}, {b.hi:+.2f}] | "
                 f"{bold}{t.gain_pct:+.2f}%{bold} [{t.lo:+.2f}, {t.hi:+.2f}] |")
    return "\n".join(L)


WORST = OrderedDict([("all", "13 лучших / 12 худших (прошлый)"), ("ext1", "1 лучшая / 1 худшая"),
                     ("tight_worst", "tight bbox GT / 1 худшая")])


def tight_worst_coco(pk: Picks, args) -> pd.DataFrame | None:
    """Table 6: on COCO half B, the tight-box-minus-worst-box vector against the
    full-halves one (and the best-minus-worst one), under every way of
    choosing alpha."""
    vs = [v for v in WORST if v in pk.fitted]
    if "tight_worst" not in vs or "COCO_B" not in set(pk.D.table):
        return None
    base = pk.on("COCO_B", "all")
    fixed = [a for a in pk.meta["fixed_alphas"] if a in pk.alphas]
    methods = [f"α = {a:g}" for a in fixed] + ["α* (подобрана на обучении)", "регрессор α, линейная",
                                                "регрессор α, GBM"]
    out = []
    for m in methods:
        P = base[["image_name", "group", "undef_iou"]].rename(columns={"undef_iou": "undef"})
        for v in vs:
            P[v] = pk.on("COCO_B", v, base, method=m).pick.to_numpy()
        T = compare_blocks(OrderedDict([("COCO_B", P)]), vs, "all", args)
        out.append(T.assign(method=m))
    T = pd.concat(out, ignore_index=True)
    for v in vs:
        T[f"cos_{v}"] = pk.cos(v, "all")
        T[f"norm_{v}"] = pk.norm_ratio(v, "all")
    return T


def md_table6(T: pd.DataFrame) -> str:
    vs = [v for v in WORST if v in T.columns]
    L = ["| выбор α | вектор (позитив / негатив) | bottom50 | top50 | к прошлому, b50 [95% CI] "
         "| к прошлому, t50 [95% CI] |", "|" + "---|" * 6]
    first = T.iloc[0]
    for m, B in T.groupby("method", sort=False):
        s = B.set_index("group")
        b, t = s.loc["bottom50"], s.loc["top50"]
        if m == T.method.iloc[0]:
            L.append(f"| — | undef | {b.undef:.4f} | {t.undef:.4f} | | |")
        for i, v in enumerate(vs):
            gb = gt = "—"
            if v != "all":
                gb = f"{b[v + '_gain']:+.2f}% [{b[v + '_lo']:+.2f}, {b[v + '_hi']:+.2f}]"
                gt = f"{t[v + '_gain']:+.2f}% [{t[v + '_lo']:+.2f}, {t[v + '_hi']:+.2f}]"
            L.append(f"| {m if i == 0 else ''} | {WORST[v]} | {b[v]:.4f} ({b[v + '_pct']:+.2f}%) | "
                     f"{t[v]:.4f} ({t[v + '_pct']:+.2f}%) | {gb} | {gt} |")
    geo = ", ".join(f"{WORST[v]}: cos {first['cos_' + v]:+.2f}, ‖v‖ ×{first['norm_' + v]:.2f}"
                    for v in vs if v != "all")
    return "\n".join(L) + f"\n\nГеометрия против прошлого вектора — {geo}."


def prev_run_diff(vdir: Path, prev: Path):
    """Max |IoU difference| over the table_exp12 rows two runs share."""
    if not (prev / "table_exp12.csv").is_file():
        return None
    a = pd.read_csv(vdir / "table_exp12.csv")
    b = pd.read_csv(prev / "table_exp12.csv")
    m = a.merge(b, on=["block", "method", "group"], suffixes=("", "_prev"))
    return float((m.iou - m.iou_prev).abs().max()) if len(m) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(a) -> int:
    """Every part whose inputs the run has; the rest is skipped and said so."""
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vdir, sdir = Path(a.variants_dir), Path(a.selector_dir)
    method_label = {"регрессор α, GBM": "выбор α по боксу: GBM-регрессор выигрыша",
                    "регрессор α, линейная": "выбор α по боксу: линейный регрессор выигрыша",
                    "α* (подобрана на обучении)": "одна α*, подобранная на обучении"}.get(a.method, a.method)
    skipped, checks, check2 = [], {}, {}

    F = figure_data(pd.read_csv(vdir / "table_exp12.csv"), a.method)
    if len(F) and F.panel.nunique():
        F.to_csv(out / "fig_annotations.csv", index=False)
        plot_figure(F, out / "fig_annotations.png", method_label)
    else:
        F, _ = None, skipped.append("1 (нет средних по натуральным / медицинским наборам)")

    pk = Picks(vdir, a.method)
    T2 = None
    try:
        T2, check2, n_draws, n_sub = own_vs_coco_sub(pk, vdir, a)
        T2.to_csv(out / "table_own_vector.csv", index=False)
    except SystemExit as e:
        skipped.append(f"2 ({e})")
    T3 = None
    if (sdir / "per_case.csv").is_file():
        T3, checks["table3_vs_selector_table_max_abs"] = table3(sdir, a)
        T3.to_csv(out / "table_catboost.csv", index=False)
    else:
        skipped.append(f"3 (нет {sdir / 'per_case.csv'})")
    T4, T5, T6 = tight_vs_base(pk, a), universal(pk, vdir, a), tight_worst_coco(pk, a)
    for name, T in (("table_tight.csv", T4), ("table_universal.csv", T5), ("table_tight_worst_coco.csv", T6)):
        if T is not None:
            T.to_csv(out / name, index=False)
    own_err = [v for k, v in check2.items() if not k.startswith("own_COCO") and v is not None]
    checks["table2_own_vs_table_exp3_max_abs"] = max(own_err) if own_err else None
    checks["own_COCO_vs_coco_draw0_max_abs"] = check2.get("own_COCO_vs_draw0_max_abs")
    if a.prev_variants_dir:
        checks["table_exp12_vs_prev_run_max_abs"] = prev_run_diff(vdir, Path(a.prev_variants_dir))
    (out / "checks.json").write_text(json.dumps({**checks, "table2_detail": check2, "skipped": skipped}, indent=2),
                                     encoding="utf-8")
    bad = [k for k, v in checks.items() if v is not None and v > 1e-6]
    if bad:
        print(f"CHECK FAILED: the report does not reproduce its sources: {checks}", file=sys.stderr)
        return 1

    md = ["# Steering SAM-1 на user study", "",
          "Steering: к токенам декодера SAM-1 vit_b после `L2.norm2` (LayerNorm `norm2` второго two-way блока, "
          "после cross-attention «токены → картинка») прибавляется вектор h' = h + α·v на всех 7 токенах "
          "(IoU, 4 mask, 2 prompt). Вектор — разность средних хороших и плохих разметок внутри картинки, "
          "усреднённая по обучающим картинкам. α выбирается по боксу GBM-регрессором выигрыша IoU "
          "(признаки без GT), обученным на картинках вектора. Группы — 12 худших (bottom50) и 13 лучших (top50) "
          "из 25 разметок картинки по undef IoU. CI — 95%, бутстреп по картинкам.", ""]
    if F is not None:
        fig_rows = ["| набор | разметок на картинку | bottom50 | top50 |", "|---|---|---|---|"]
        for panel, P in F.groupby("panel", sort=False):
            nm = "натуральные" if panel == "natural" else "медицина"
            u = P.groupby("group").undef.first()
            fig_rows.append(f"| {nm} | undef | {u['bottom50']:.4f} | {u['top50']:.4f} |")
            for var, _, tick in K_POINTS:
                s = P[P.variant == var].set_index("group")
                fig_rows.append(f"| {nm} | {tick.splitlines()[0]} | {s.loc['bottom50', 'iou']:.4f} "
                                f"({s.loc['bottom50', 'pct']:+.2f}%) | {s.loc['top50', 'iou']:.4f} "
                                f"({s.loc['top50', 'pct']:+.2f}%) |")
        md += ["## 1. Сколько разметок на картинку нужно для вектора", "",
               "Вектор по 25 картинкам COCO A из k лучших и k худших разметок каждой картинки; тест — COCO "
               "(половина B), Berkeley, GrabCut и ACDC, BUID, MedScribble целиком; среднее по наборам с равным весом.",
               "", "![IoU от числа разметок](fig_annotations.png)", "", "\n".join(fig_rows), ""]
    if T2 is not None:
        md += [f"## 2. Свой вектор на набор против вектора COCO по тем же {n_sub} картинкам", "",
               f"Свой вектор — {n_sub} случайных картинок набора со всеми разметками; COCO-{n_sub} — {n_sub} случайных "
               f"картинок половины A COCO со всеми разметками, {n_draws} разных выборок, в столбцах COCO-{n_sub} — "
               f"среднее IoU по выборкам (ожидаемое качество вектора COCO из {n_sub} картинок). Всё на одних и тех же "
               "тестовых картинках набора (40; COCO — 25 картинок половины B). Проценты в столбцах методов — к undef; "
               f"«свой к COCO-{n_sub}» — прирост IoU своего вектора в % от IoU COCO-{n_sub}, парный бутстреп по "
               f"картинкам; последний столбец — тот же прирост против каждой выборки COCO-{n_sub} отдельно. cos — по "
               f"всему тензору 7 × 256, среднее по выборкам. ¹ Для COCO свой вектор — это выборка 0 COCO-{n_sub}, "
               f"поэтому COCO-{n_sub} здесь — среднее по остальным выборкам (1–{n_draws - 1}), а разброс в средних "
               f"по наборам — по выборкам 1–{n_draws - 1}.", "",
               md_table2(T2, n_draws, n_sub), ""]
    if T3 is not None:
        md += ["## 3. CatBoost-селектор без и со steering-кандидатами", "",
               "Пул без steering — undef, 16 BoN и 11 точек grad; со steering — плюс 8 кандидатов: вектор COCO A "
               "(25 картинок) при α = 1, 2, 3, 4, 6, 8, 12, 16; какую α (или не-steering кандидата) взять, решает "
               "сам CatBoost (α — признак кандидата). CatBoost YetiRank, 5 фолдов по картинкам, все 12 500 разметок, "
               "каждая оценена вне фолда. Проценты в столбцах — к undef; «выигрыш» — IoU со steering в % от IoU без "
               "steering, парный бутстреп.", "", md_table3(T3), ""]
    if T4 is not None:
        md += ["## 4. Tight bbox GT как позитивный пример", "",
               "Все векторы — по 25 картинкам COCO A. Позитив «tight bbox GT» — состояние декодера на tight bbox "
               "маски GT (идеальный бокс). Тест — все тестовые наборы прогона. Проценты у IoU — к undef; "
               "«к базовому» — прирост IoU в % от базового вектора (13 лучших / 12 худших), парный бутстреп по "
               "картинкам.", "", md_table4(T4), ""]
    if T5 is not None:
        md += ["## 5. Один вектор для всех наборов", "",
               "На тестовых картинках каждого набора (как в п. 2): свой вектор набора (10 картинок); общий вектор — "
               "те же 10-картиночные выборки всех 6 наборов вместе (60 картинок, каждая картинка и каждый набор с "
               "равным весом), тестовые картинки ни одного набора в него не входят; общий без этого набора — "
               "5 × 10 картинок других наборов (набор для вектора не виден); вектор COCO A (25 картинок). α по боксу "
               "выбирает GBM-регрессор, обученный на картинках вектора. Проценты у IoU — к undef; «к своему» — "
               "прирост в % от своего вектора, парный бутстреп; cos и отношение норм — со своим вектором набора.", "",
               md_long(T5, UNIVERSAL, "own", "своему", "cos со своим"), ""]
    if T6 is not None:
        md += ["## 6. COCO: tight bbox GT против одной худшей разметки", "",
               "Векторы по 25 картинкам COCO A, тест — 25 картинок половины B (625 разметок). «tight bbox GT / "
               "1 худшая» — разность состояния декодера на tight bbox маски GT и на худшей разметке картинки; "
               "«прошлый» — 13 лучших / 12 худших. Для каждого способа выбора α (фиксированные, одна α*, "
               f"регрессоры; сетка α = {', '.join(f'{x:g}' for x in pk.alphas)}) — IoU (% к undef) и прирост к "
               "прошлому вектору в %, парный бутстреп по картинкам.", "", md_table6(T6), ""]
    if skipped:
        md += ["Не посчитано в этом прогоне: " + "; ".join(skipped) + ".", ""]
    text = "\n".join(md)
    (out / "report.md").write_text(text, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, "replace").decode(enc))
    print(f"\nchecks: {checks}\nwrote {out}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants_dir", default="exp_res/steer_variants_v2")
    p.add_argument("--prev_variants_dir", default="exp_res/steer_variants",
                   help="an earlier steer_variants run whose table_exp12 rows this one must reproduce ('' = no check)")
    p.add_argument("--selector_dir", default="results/selector_steer")
    p.add_argument("--method", default="регрессор α, GBM",
                   choices=["регрессор α, GBM", "регрессор α, линейная", "α* (подобрана на обучении)"])
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="exp_res/steer_userstudy_report")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
