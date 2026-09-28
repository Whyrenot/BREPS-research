"""
selector_policies.py
====================
Turning a per-candidate score into a decision, and a decision into a number.

Two kinds of policy live here. `argmax` is the whole story for the pooled
choices (T2, T3): every candidate already exists, so picking is free and only
the ranking matters. The stop rules are for T1, where the candidates arrive one
at a time and the point is to not compute all of them -- so they return a cost
as well as an IoU, and a policy that scores well at twice the forwards has not
won anything.

Reporting. Every delta is paired per case and then AVERAGED WITHIN AN IMAGE
before the standard error is taken. The 25 annotations of one image share an
object and are not independent draws; treating them as such would shrink the
error bars by up to 5x. The z is over images, n=500 (or 100 on the holdout).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = ("all", "weak", "mid", "strong")


# --------------------------------------------------------------------------
# reshaping a long pool into (case, slot) matrices
# --------------------------------------------------------------------------
class CaseGrid:
    """A pool with a fixed number of candidates per case, as dense matrices.

    Every builder in selector_data emits equal-size, contiguous, correctly
    ordered pools, so this is a reshape rather than a pivot -- which is what
    makes the stop rules a handful of vectorised passes instead of 12500 Python
    loops.
    """

    def __init__(self, d: pd.DataFrame):
        case = d.case.to_numpy()
        chg = np.r_[True, case[1:] != case[:-1]]
        start = np.flatnonzero(chg)
        size = np.diff(np.r_[start, len(case)])
        if size.min() != size.max():
            raise ValueError(f"ragged pools: {size.min()}..{size.max()} candidates")
        self.n, self.t = len(start), int(size[0])
        self.cases = case[start]
        self.iou = d.target_iou.to_numpy().reshape(self.n, self.t)
        self.cost = (d.n_batched_fwd.to_numpy().reshape(self.n, self.t)
                     if "n_batched_fwd" in d else np.ones((self.n, self.t)))

    def col(self, d: pd.DataFrame, name: str) -> np.ndarray:
        return d[name].to_numpy().reshape(self.n, self.t)

    # ---- policies -------------------------------------------------------
    def argmax(self, s: np.ndarray):
        k = np.nanargmax(np.where(np.isnan(s), -np.inf, s), axis=1)
        r = np.arange(self.n)
        return self.iou[r, k], self.cost[r, -1], k

    def oracle(self):
        return self.iou.max(axis=1), self.cost[:, -1], self.iou.argmax(axis=1)

    def restricted(self, s: np.ndarray, sel: np.ndarray):
        """Argmax over a subset of the slots. Passing the true IoU as `s` makes
        this the oracle over that subset -- which is how "could you win just by
        choosing the METHOD, never mind which of the 16 bon draws" gets a row in
        the same table as everything else."""
        v = np.where(sel, np.where(np.isnan(s), -np.inf, s), -np.inf)
        k = v.argmax(axis=1)
        r = np.arange(self.n)
        return self.iou[r, k], self.cost[r, -1], k

    def argmax_guarded(self, s: np.ndarray, inc: np.ndarray, tau: float):
        """Argmax, but only switch away from the incumbent by a real margin.

        Every one of these pools contains near-duplicate masks whose true IoUs
        differ in the 4th decimal, so an unguarded argmax spends most of its
        switches on coin flips -- each one a chance to lose, and none of them a
        chance to win much. Scores are z-scored inside the pool first, so tau
        is in pool sigmas and means the same thing for a YetiRank score and a
        logit. tau=0 is plain argmax.
        """
        s = np.where(np.isnan(s), -np.inf, s)
        fin = np.where(np.isfinite(s), s, np.nan)
        mu = np.nanmean(fin, axis=1, keepdims=True)
        sd = np.nanstd(fin, axis=1, keepdims=True)
        z = np.where(sd > 0, (s - mu) / np.where(sd > 0, sd, 1.0), 0.0)
        r = np.arange(self.n)
        k = np.argmax(z, axis=1)
        k = np.where(z[r, k] - z[r, inc] > tau, k, inc)
        return self.iou[r, k], self.cost[r, -1], k

    def stop_patience(self, s: np.ndarray, patience: float):
        """Walk in order; halt once the score has fallen more than `patience`
        below its running best, and return the best checkpoint seen."""
        s = np.where(np.isnan(s), -np.inf, s)
        best_i = np.zeros(self.n, dtype=int)
        best_v = s[:, 0].copy()
        done = np.zeros(self.n, dtype=bool)
        stop_t = np.full(self.n, self.t - 1, dtype=int)
        for t in range(1, self.t):
            live = ~done
            up = live & (s[:, t] > best_v)
            best_i[up], best_v[up] = t, s[up, t]
            halt = live & ~up & (best_v - s[:, t] > patience)
            stop_t[halt] = t
            done |= halt
        r = np.arange(self.n)
        return self.iou[r, best_i], self.cost[r, stop_t], best_i

    def stop_prob(self, s: np.ndarray, p_cont: np.ndarray, thr: float):
        """Halt the first time the continue-probability drops below `thr`, and
        return the best checkpoint seen so far according to `s`."""
        s = np.where(np.isnan(s), -np.inf, s)
        p = np.where(np.isnan(p_cont), 0.0, p_cont)
        best_i = np.zeros(self.n, dtype=int)
        best_v = s[:, 0].copy()
        done = np.zeros(self.n, dtype=bool)
        stop_t = np.full(self.n, self.t - 1, dtype=int)
        for t in range(self.t):
            live = ~done
            if t > 0:
                up = live & (s[:, t] > best_v)
                best_i[up], best_v[up] = t, s[up, t]
            if t == self.t - 1:
                break
            halt = live & (p[:, t] < thr)
            stop_t[halt] = t
            done |= halt
        r = np.arange(self.n)
        return self.iou[r, best_i], self.cost[r, stop_t], best_i


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def image_clustered_z(delta: np.ndarray, img: np.ndarray):
    """Mean delta and its z, clustering on the image."""
    if len(delta) == 0:
        return np.nan, np.nan
    per = pd.Series(delta).groupby(img).mean()
    se = per.std(ddof=1) / np.sqrt(max(1, per.size))
    m = float(np.mean(delta))
    return m, (abs(m) / se if se and se > 0 else np.nan)


def summarize(results: dict, ref: str, oracle: str, img: np.ndarray,
              grp: np.ndarray, mask: np.ndarray, task: str, split: str
              ) -> pd.DataFrame:
    """One row per (group, policy).

    `results[name] = (iou, cost, slot)`. `ref` is the incumbent the deltas are
    measured against -- today's behaviour, not the weakest baseline -- and
    `oracle` bounds what any selector over the same pool could have reached, so
    "share of oracle" says how much of the available headroom this score bought.
    """
    rows = []
    for g in GROUPS:
        m = mask if g == "all" else (mask & (grp == g))
        if not m.any():
            continue
        ri = results[ref][0][m]
        oi = results[oracle][0][m]
        gap = float(oi.mean() - ri.mean())
        for name, (iou, cost, slot) in results.items():
            v = iou[m]
            d, z = image_clustered_z(v - ri, img[m])
            rows.append({
                "task": task, "split": split, "group": g, "n": int(m.sum()),
                "policy": name, "iou": float(v.mean()), "d_vs_ref": d,
                "z": z, "share_of_oracle": (d / gap if gap > 1e-12 else np.nan),
                "cost": float(np.mean(cost[m])),
                # "hit" not "top-1": the bon pool is full of duplicate masks,
                # so matching the oracle's INDEX would punish a policy for
                # picking an equally good twin. What counts is the IoU.
                "hit": float(np.mean(v >= oi - 1e-9)),
            })
    return pd.DataFrame(rows)


def render(tab: pd.DataFrame, ref: str, show_cost: bool) -> str:
    """The table as the terminal should see it: groups down, policies within."""
    out = []
    w = max(len(p) for p in tab.policy.unique())
    head = f"{'group':>6} | {'policy':<{w}} | {'IoU':>7} | {'vs ref':>8} {'z':>5} | {'share':>6} | {'hit':>5}"
    if show_cost:
        head += f" | {'fwd':>5}"
    for g in GROUPS:
        s = tab[tab.group == g]
        if s.empty:
            continue
        out.append("")
        out.append(head)
        out.append("-" * len(head))
        for _, r in s.iterrows():
            is_ref = r.policy == ref
            dv = f"{'ref':>14}" if is_ref else f"{r.d_vs_ref:>+8.4f} {r.z:>5.1f}"
            sh = "     -" if is_ref or not np.isfinite(r.share_of_oracle) else f"{r.share_of_oracle:>6.1%}"
            line = (f"{r.group:>6} | {r.policy:<{w}} | {r.iou:>7.4f} | {dv} | "
                    f"{sh} | {r.hit:>5.1%}")
            if show_cost:
                line += f" | {r.cost:>5.1f}"
            out.append(line)
    return "\n".join(out)
