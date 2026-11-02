"""Spurious heterogeneity under omitted curvature (DGP-B) and varying dispersion (DGP-C).

Neither process has random parameters, so any parameter declared random is spurious.
DGP-B targets x1 (convex effect), DGP-C targets x11 (dispersion depends on x11).

    python scripts/analyze_bc.py --fig ../manuscript/figures --tab ../manuscript/tables
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402  (shared style, palette and helpers)

import matplotlib.pyplot as plt  # noqa: E402

TARGET = {"B": 0, "C": 10}
TARGET_LABEL = {"B": "x1", "C": "x11"}


def load(dgp: str, root: Path):
    recs = [json.load(open(f)) for f in sorted(glob.glob(str(root / f"dgp{dgp}" / "raw" / "*.json")))]
    for sub in ("raw_m23", "raw_m6"):
        extra = root / f"dgp{dgp}" / sub
        if not extra.exists():
            continue
        ex = {(e["cell"], e["rep"]): e for e in (json.load(open(f)) for f in glob.glob(str(extra / "*.json")))}
        for r in recs:
            e = ex.get((r["cell"], r["rep"]))
            if e:
                r.update({m: e[m] for m in ("M2", "M3", "M6") if m in e})
    return recs


def metrics(recs, dgp, methods):
    j0 = TARGET[dgp]
    rows = []
    for r in recs:
        for m in methods:
            s = np.array(r[m]["states"])
            row = dict(dgp=dgp, n=r["n"], T=r["T"], method=m,
                       target_R=float(s[j0] == 2), any_R=float((s == 2).any()),
                       n_R=float((s == 2).sum()))
            if "P" in r[m]:
                row["target_PR"] = float(np.array(r[m]["P"])[j0, 2])
            if m in ("M5", "M6") and dgp == "B" and "quad_states" in r[m]:
                row["quad_x1_selected"] = float(r[m]["quad_states"][0] > 0)
            if "P_disp" in r[m]:
                pdisp = np.array(r[m]["P_disp"])
                row["target_disp"] = float(pdisp[j0] > 0.5)
                row["target_Pdisp"] = float(pdisp[j0])
                row["other_disp"] = float((np.delete(pdisp, j0) > 0.5).any())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    cols = [c for c in ("target_R", "any_R", "n_R", "target_PR", "quad_x1_selected", "target_disp",
                        "target_Pdisp", "other_disp") if c in df]
    g = df.groupby(["dgp", "n", "T", "method"])
    mean = g[cols].mean()
    se = g[cols].std(ddof=1) / np.sqrt(g.size().values[:, None])
    se.columns = [c + "_se" for c in cols]
    out = pd.concat([mean, se], axis=1).reset_index()
    out["reps"] = g.size().values
    return out


def figure(summ, methods, figdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6), sharey=True)
    rows = [("target_R", "Target covariate declared random"), ("any_R", "At least one parameter declared random")]
    cols = [("B", "DGP-B: convex effect of x1"), ("C", "DGP-C: dispersion depends on x11")]
    offs = dict(zip(methods, np.linspace(-0.24, 0.24, len(methods)) if len(methods) > 1 else [0.0]))
    for r_i, (col, ylab) in enumerate(rows):
        for c_i, (dgp, title) in enumerate(cols):
            ax = axes[r_i, c_i]
            sub = summ[summ.dgp == dgp]
            Ts = sorted(sub["T"].unique())
            for m in methods:
                sty = A.METHODS[m]
                sm = sub[sub.method == m].set_index("T")
                if sm.empty:
                    continue
                x = np.arange(len(Ts)) + offs[m]
                yv = np.array([sm.loc[T, col] for T in Ts])
                ye = 1.96 * np.array([sm.loc[T, col + "_se"] for T in Ts])
                ax.errorbar(x, yv, yerr=ye, fmt="none", ecolor=sty["color"], elinewidth=1.1, zorder=2)
                ms = 6.6 if sty["marker"] in "^vP" else 5.6
                ax.plot(x, yv, linestyle="none", marker=sty["marker"], markersize=ms, color=sty["color"],
                        markeredgecolor=A.SURFACE, markeredgewidth=1.4, label=sty["label"], zorder=3)
            ax.set_xticks(range(len(Ts)))
            ax.set_xticklabels([("Cross-section\nT = 1" if T == 1 else f"Panel\nT = {T}") for T in Ts])
            ax.set_xlim(-0.5, len(Ts) - 0.5)
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
            ax.grid(axis="x", visible=False)
            A._despine(ax)
            if r_i == 0:
                ax.set_title(title, color=A.INK)
            if c_i == 0:
                ax.set_ylabel(ylab.replace(" declared", "\ndeclared").replace("parameter ", "parameter\n"))
    h, lab = axes[0, 0].get_legend_handles_labels()
    nc = 2 if len(methods) == 4 else 3
    fig.legend(h, lab, loc="upper center", ncol=nc, bbox_to_anchor=(0.5, 1.06 if len(methods) > 3 else 1.035),
               handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(methods) > 3 else 0.95))
    A._save(fig, figdir, "dgpBC_spurious")


def table(summ, methods, tabdir: Path):
    L = []
    for m in methods:
        vals = []
        for dgp in ("B", "C"):
            for T in (1, 5):
                x = summ[(summ.dgp == dgp) & (summ["T"] == T) & (summ.method == m)]
                vals.append(f"{x.target_R.iloc[0]:.2f} & {x.any_R.iloc[0]:.2f}" if len(x) else "-- & --")
        lab = A.METHODS[m]["label"].replace(" ", "\\\\ ", 1)
        L.append(f"\\shortstack[l]{{{lab}}} & " + " & ".join(vals) + " \\\\")
    tabdir.mkdir(parents=True, exist_ok=True)
    (tabdir / "dgpBC_spurious_rows.tex").write_text("\n\\addlinespace\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--fig", default="../manuscript/figures")
    ap.add_argument("--tab", default="../manuscript/tables")
    a = ap.parse_args()
    root = Path(a.root)
    frames, methods = [], None
    for dgp in ("B", "C"):
        recs = load(dgp, root)
        if not recs:
            print(f"DGP-{dgp}: no results yet")
            continue
        present = [m for m in A.METHODS if all(m in r for r in recs)]
        methods = present if methods is None else [m for m in methods if m in present]
        frames.append((dgp, recs))
    if not frames:
        return
    df = pd.concat([metrics(recs, dgp, methods) for dgp, recs in frames])
    summ = summarize(df)
    out = root / "dgpBC"
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "spurious_summary.csv", index=False)
    figure(summ, methods, Path(a.fig))
    table(summ, methods, Path(a.tab))
    pd.set_option("display.width", 200)
    show = [c for c in ("dgp", "n", "T", "method", "reps", "target_R", "any_R", "n_R", "target_PR", "quad_x1_selected",
                        "target_disp", "target_Pdisp", "other_disp") if c in summ]
    print(summ[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
