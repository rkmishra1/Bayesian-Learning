"""Aggregate simulation output into metrics, LaTeX tables and figures.

    python scripts/analyze.py --raw results/dgpA/raw --out results/dgpA \
        --fig ../manuscript/figures --tab ../manuscript/tables
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpsel.summaries import fdr_select  # noqa: E402

# ----------------------------------------------------------------------------- style
# Reference palette from the dataviz skill. The five method colours were validated for
# colour-vision deficiency in the drawing order M1..M5 (adjacent pairs). The proposed
# methods get the two strongest hues, and every method also has its own marker shape.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
METHODS = {
    "M1": dict(label="M1 Sequential SML", color="#1baf7a", marker="D"),
    "M2": dict(label="M2 Sequential Bayes (DIC)", color="#eda100", marker="^"),
    "M3": dict(label="M3 BIC search", color="#e87ba4", marker="v"),
    "M4": dict(label="M4 Proposed, linear mean", color="#2a78d6", marker="o"),
    "M5": dict(label="M5 Proposed, flexible mean", color="#eb6834", marker="s"),
    "M6": dict(label="M6 Proposed, mean and dispersion", color="#6250d6", marker="P"),
}
ACTIVE: list[str] = list(METHODS)  # methods present in every record, set in main()
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATE_NAME = {0: "E", 1: "F", 2: "R"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "pdf.fonttype": 42,
})


def _despine(ax, left=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


def _save(fig, figdir: Path, name: str):
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- load
def load(raw: str, raw_extra: str | None = None) -> list[dict]:
    """Load main records and merge extra methods (M2, M3) stored per (cell, replication)."""
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{raw}/*.json"))]
    if not recs:
        raise SystemExit(f"no results in {raw}")
    for d in (raw_extra or "").split(","):
        if not d:
            continue
        extra = {(e["cell"], e["rep"]): e for e in (json.load(open(f)) for f in glob.glob(f"{d}/*.json"))}
        for r in recs:
            e = extra.get((r["cell"], r["rep"]))
            if e is not None:
                for m in ("M2", "M3", "M6"):
                    if m in e:
                        r[m] = e[m]
    return recs


def design_label(n, T):
    return f"n={n}, T={T}"


def rep_metrics(recs) -> pd.DataFrame:
    rows = []
    for r in recs:
        t = np.array(r["true_states"])
        for m in ACTIVE:
            s = np.array(r[m]["states"])
            nR = int((s == 2).sum())
            rows.append(dict(
                n=r["n"], T=r["T"], rep=r["rep"], method=m,
                exact=float(np.all(s == t)),
                acc3=float(np.mean(s == t)),
                R_tpr=float(np.mean(s[t == 2] == 2)),
                R_fpr=float(np.mean(s[t != 2] == 2)),
                R_fdp=float(((s == 2) & (t != 2)).sum() / max(nR, 1)),
                R_on_fixed=float(np.mean(s[t == 1] == 2)),
                incl_tpr=float(np.mean(s[t > 0] > 0)),
                incl_fpr=float(np.mean(s[t == 0] > 0)),
                seconds=float(r[m]["seconds"]),
            ))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["exact", "acc3", "R_tpr", "R_fpr", "R_fdp", "R_on_fixed", "incl_tpr", "incl_fpr", "seconds"]
    g = df.groupby(["n", "T", "method"])
    mean = g[cols].mean()
    se = g[cols].std(ddof=1) / np.sqrt(g.size().values[:, None])
    se.columns = [c + "_se" for c in cols]
    out = pd.concat([mean, se], axis=1).reset_index()
    out["reps"] = g.size().values
    return out


def estimation(recs) -> pd.DataFrame:
    """RMSE and 95% interval coverage for the true random parameters (mean and |sigma|)."""
    rows = []
    for r in recs:
        tb, ts = np.array(r["true_beta"]), np.array(r["true_sigma"])
        J = np.nonzero(ts > 0)[0]
        for m in ACTIVE:
            e = r[m]
            if m in ("M1", "M3"):  # Wald intervals from the final ML fit
                b, s = np.array(e["beta"]), np.abs(np.array(e["sigma"]))
                bse = np.nan_to_num(np.array(e["beta_se"], dtype=float))
                sse = np.nan_to_num(np.array(e["sigma_se"], dtype=float))
                blo, bhi = b - 1.96 * bse, b + 1.96 * bse
                slo, shi = np.maximum(s - 1.96 * sse, 0), s + 1.96 * sse
            elif m == "M2":  # posterior intervals from the final Bayesian fit
                b, s = np.array(e["beta"]), np.array(e["sigma"])
                blo, bhi = np.array(e["beta_lo"]), np.array(e["beta_hi"])
                slo, shi = np.array(e["sigma_lo"]), np.array(e["sigma_hi"])
            else:
                b, s = np.array(e["beta_mean"]), np.array(e["sigma_mean"])
                blo, bhi = np.array(e["beta_lo"]), np.array(e["beta_hi"])
                slo, shi = np.array(e["sigma_lo"]), np.array(e["sigma_hi"])
            for j in J:
                rows.append(dict(
                    n=r["n"], T=r["T"], method=m, j=int(j),
                    b_err=b[j] - tb[j], s_err=s[j] - ts[j],
                    b_cov=float(blo[j] <= tb[j] <= bhi[j]),
                    s_cov=float(slo[j] <= ts[j] <= shi[j]),
                ))
    d = pd.DataFrame(rows)
    g = d.groupby(["n", "T", "method"])
    return pd.DataFrame({
        "beta_rmse": g["b_err"].apply(lambda x: np.sqrt(np.mean(x**2))),
        "beta_cov": g["b_cov"].mean(),
        "sigma_rmse": g["s_err"].apply(lambda x: np.sqrt(np.mean(x**2))),
        "sigma_cov": g["s_cov"].mean(),
    }).reset_index()


def per_covariate(recs) -> pd.DataFrame:
    rows = []
    for r in recs:
        t = r["true_states"]
        for m in ACTIVE:
            for j, s in enumerate(r[m]["states"]):
                rows.append(dict(n=r["n"], T=r["T"], method=m, j=j, truth=t[j],
                                 sel_R=float(s == 2), sel_incl=float(s > 0)))
    return pd.DataFrame(rows).groupby(["n", "T", "method", "j", "truth"])[["sel_R", "sel_incl"]].mean().reset_index()


# ----------------------------------------------------------------------------- figures
def fig_detection(summ: pd.DataFrame, figdir: Path):
    designs = summ[["n", "T"]].drop_duplicates().sort_values(["n", "T"]).values.tolist()
    panels = [
        ("R_tpr", "(a) Random parameters detected", "Share of true random parameters\nselected as random"),
        ("R_on_fixed", "(b) Fixed effects declared random", "Share of true fixed effects\nselected as random"),
        ("incl_fpr", "(c) Null covariates included", "Share of null covariates\nselected as fixed or random"),
        ("exact", "(d) Whole specification recovered", "Share of replications with\nall 20 states correct"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
    offs = dict(zip(ACTIVE, np.linspace(-0.3, 0.3, len(ACTIVE)) if len(ACTIVE) > 1 else [0.0]))
    for ax, (col, title, ylab) in zip(axes.ravel(), panels):
        for m in ACTIVE:
            sty = METHODS[m]
            sub = summ[summ.method == m].set_index(["n", "T"])
            x = np.arange(len(designs)) + offs[m]
            yv = np.array([sub.loc[tuple(d), col] for d in designs])
            ye = np.array([sub.loc[tuple(d), col + "_se"] for d in designs]) * 1.96
            ax.errorbar(x, yv, yerr=ye, fmt="none", ecolor=sty["color"], elinewidth=1.1, capsize=0, alpha=0.9, zorder=2)
            ms = 6.6 if sty["marker"] in "^vP" else 5.6  # triangles and plus read smaller at equal size
            ax.plot(x, yv, linestyle="none", marker=sty["marker"], markersize=ms, color=sty["color"],
                    markeredgecolor=SURFACE, markeredgewidth=1.4, label=sty["label"], zorder=3)
        ax.set_title(title, color=INK)
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, len(designs) - 0.5)
        _despine(ax)
    for ax in axes[1]:
        ax.set_xticks(range(len(designs)))
        ax.set_xticklabels([f"n = {n}\nT = {T}" for n, T in designs])
    h, lab = axes[0, 0].get_legend_handles_labels()
    ncol = 3 if len(ACTIVE) <= 3 else 3
    fig.legend(h, lab, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.06), handletextpad=0.3,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if len(ACTIVE) > 3 else 0.96))
    _save(fig, figdir, "dgpA_detection")


def fig_heatmap(pc: pd.DataFrame, figdir: Path):
    designs = pc[["n", "T"]].drop_duplicates().sort_values(["n", "T"]).values.tolist()
    truth = pc.drop_duplicates("j").set_index("j")["truth"]
    order = list(truth[truth == 2].index) + list(truth[truth == 1].index) + list(truth[truth == 0].index)
    ylabels = [f"x{j + 1}  ({STATE_NAME[truth[j]]})" for j in order]
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    nm = len(ACTIVE)
    fig, axes = plt.subplots(2, nm, figsize=(6.6 if nm <= 3 else (7.0 if nm <= 5 else 7.6), 7.4), sharey=True,
                             squeeze=False)
    rows = [("sel_R", "Selected as random"), ("sel_incl", "Selected as fixed or random")]
    for r_i, (col, rlab) in enumerate(rows):
        for c_i, m in enumerate(ACTIVE):
            ax = axes[r_i, c_i]
            sub = pc[pc.method == m].set_index(["j", "n", "T"])[col]
            M = np.array([[sub.loc[(j, n, T)] for (n, T) in designs] for j in order])
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(designs)))
            ax.set_xticklabels([f"{n}\n{T}" for n, T in designs], fontsize=6.5 if nm > 3 else 7.5)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(ylabels, fontsize=7.5)
            for b in (2.5, 6.5):  # separators between truth groups
                ax.axhline(b, color=SURFACE, linewidth=2.0)
            for b in np.arange(0.5, len(designs) - 0.5):
                ax.axvline(b, color=SURFACE, linewidth=1.0)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r_i == 0:
                lab = METHODS[m]["label"]
                lab = lab.replace("Proposed, ", "Proposed,\n").replace("Sequential ", "Sequential\n").replace("BIC search", "BIC\nsearch").replace("mean and dispersion", "mean and\ndispersion")
                ax.set_title(lab, fontsize=7.5 if nm > 3 else 8.5)
            if c_i == 0:
                ax.set_ylabel(rlab, color=INK)
        axes[r_i, 0].annotate("n\nT", xy=(-0.02, -0.02), xycoords="axes fraction", ha="right", va="top",
                              fontsize=7.5, color=MUTED)
    cax = fig.add_axes([0.30, 0.035, 0.45, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.set_label("Share of replications", color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, figdir, "dgpA_selection_heatmap")


def fig_calibration(recs, figdir: Path):
    designs = sorted({(r["n"], r["T"]) for r in recs})
    fig, axes = plt.subplots(1, len(designs), figsize=(6.6, 2.15), sharey=True)
    edges = np.linspace(0, 1, 11)
    sty = METHODS["M4"]
    for ax, (n, T) in zip(axes, designs):
        p, o = [], []
        for r in recs:
            if (r["n"], r["T"]) != (n, T):
                continue
            pr = np.array(r["M4"]["P"])[:, 2]
            p.append(pr)
            o.append(np.array(r["true_states"]) == 2)
        p, o = np.concatenate(p), np.concatenate(o)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.0, zorder=1)
        for b in range(10):
            sel = idx == b
            if sel.sum() == 0:
                continue
            ax.plot(p[sel].mean(), o[sel].mean(), linestyle="none", marker=sty["marker"],
                    markersize=3.0 + 6.0 * np.sqrt(sel.sum() / len(p)), color=sty["color"],
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.set_title(f"n = {n}, T = {T}", fontsize=8.5)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_aspect("equal")
        _despine(ax)
    axes[0].set_ylabel("Observed share\nthat are random")
    fig.supxlabel("Posterior probability of being random, P(R), method M4 (bins of width 0.1; "
                  "marker area grows with the number of candidates in the bin)",
                  fontsize=8, color=INK2, y=0.0)
    fig.tight_layout()
    _save(fig, figdir, "dgpA_calibration")


STATE_RAMP = {"E": "#86b6ef", "F": "#2a78d6", "R": "#104281"}  # ordinal blue, validated
STATE_TEXT = {"E": INK, "F": SURFACE, "R": SURFACE}


def fig_state_probs(recs, figdir: Path, designs=((1000, 1), (1000, 5)), rep: int = 0):
    """Posterior state probabilities (M4) for one pre-specified replication per design."""
    pick = {(r["n"], r["T"]): r for r in recs if r["rep"] == rep and (r["n"], r["T"]) in designs}
    if len(pick) < len(designs):
        return
    truth = np.array(pick[designs[0]]["true_states"])
    order = list(np.nonzero(truth == 2)[0]) + list(np.nonzero(truth == 1)[0]) + list(np.nonzero(truth == 0)[0])
    fig, axes = plt.subplots(1, len(designs), figsize=(6.6, 4.6), sharey=True)
    gap = 0.004  # surface gap between segments, in probability units
    for ax, (n, T) in zip(axes, designs):
        r = pick[(n, T)]
        P = np.array(r["M4"]["P"])
        sel = np.array(r["M4"]["states"])
        for row, j in enumerate(order):
            left = 0.0
            for k, s in enumerate("EFR"):
                w = P[j, k]
                if w > 0:
                    ax.barh(row, max(w - gap, 0), left=left + gap / 2, height=0.62, color=STATE_RAMP[s],
                            edgecolor="none")
                    if w >= 0.16:
                        ax.text(left + w / 2, row, f"{w:.2f}", ha="center", va="center", fontsize=6.5,
                                color=STATE_TEXT[s])
                left += w
            if sel[j] != truth[j]:  # flag misclassified candidates
                ax.text(1.03, row, "✕", ha="left", va="center", fontsize=7, color=INK2,
                        transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        kind = "Cross-section" if T == 1 else "Panel"
        ax.set_title(f"{kind}: n = {n}, T = {T}", fontsize=8.5)
        for b in (2.5, 6.5):
            ax.axhline(b, color=AXIS, linewidth=0.6)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"x{j + 1}  ({STATE_NAME[truth[j]]})" for j in order], fontsize=7.5)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_RAMP[s]) for s in "EFR"]
    fig.legend(handles, ["Excluded", "Fixed", "Random"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.035), handlelength=1.2, columnspacing=1.6)
    fig.supxlabel("Posterior state probability (method M4).  Labels give the true state;  "
                  "✕ marks a candidate whose selected state is wrong.", fontsize=7.5, color=INK2, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, figdir, "dgpA_state_probabilities")


# ----------------------------------------------------------------------------- tables
def table_recovery(summ: pd.DataFrame, tabdir: Path, reps: int):
    """Rows: design blocks of six measures. Columns: methods, so methods compare side by side."""
    designs = summ[["n", "T"]].drop_duplicates().sort_values(["n", "T"]).values.tolist()
    rows = [("exact", "Exact recovery"), ("acc3", "Three-state accuracy"),
            ("R_tpr", "TPR (R)"), ("R_on_fixed", "Fixed declared R"), ("R_fdp", "FDP (R)"),
            ("incl_fpr", "Null included")]
    L = []
    for n, T in designs:
        sub = summ[(summ.n == n) & (summ["T"] == T)].set_index("method")
        n_tex = f"{n:,}".replace(",", "{,}")
        for i, (col, lab) in enumerate(rows):
            head = f"\\multirow{{{len(rows)}}}{{*}}{{\\shortstack[l]{{$n={n_tex}$\\\\ $T={T}$}}}}" if i == 0 else ""
            vals = " & ".join(f"{sub.loc[m, col]:.2f}" for m in ACTIVE)
            L.append(f"{head} & {lab} & {vals} \\\\")
        L.append("\\midrule")
    L = L[:-1]
    tabdir.mkdir(parents=True, exist_ok=True)
    (tabdir / "dgpA_recovery_rows.tex").write_text("\n".join(L) + "\n")


def table_estimation(est: pd.DataFrame, tabdir: Path):
    designs = est[["n", "T"]].drop_duplicates().sort_values(["n", "T"]).values.tolist()
    L = []
    for m in ACTIVE:
        sub = est[est.method == m].set_index(["n", "T"])
        cells = []
        for d in designs:
            x = sub.loc[tuple(d)]
            cells.append(f"{x.beta_rmse:.2f} & {x.beta_cov:.2f} & {x.sigma_rmse:.2f} & {x.sigma_cov:.2f}")
        L.append(f"{m} & " + " & ".join(cells[:2]) + " \\\\")
    L2 = []
    for m in ACTIVE:
        sub = est[est.method == m].set_index(["n", "T"])
        cells = []
        for d in designs[2:]:
            x = sub.loc[tuple(d)]
            cells.append(f"{x.beta_rmse:.2f} & {x.beta_cov:.2f} & {x.sigma_rmse:.2f} & {x.sigma_cov:.2f}")
        L2.append(f"{m} & " + " & ".join(cells) + " \\\\")
    (tabdir / "dgpA_estimation_rows_n300.tex").write_text("\n".join(L) + "\n")
    (tabdir / "dgpA_estimation_rows_n1000.tex").write_text("\n".join(L2) + "\n")


def table_rules(recs, tabdir: Path, out: Path):
    """Median probability rule versus Bayesian FDR control (q = 0.10, 0.05) for M4 and M5."""
    rows = []
    for r in recs:
        t = np.array(r["true_states"])
        for m in ("M4", "M5"):
            if m not in r:
                continue
            Pm = np.array(r[m]["P"])
            for rule, q in (("median", None), ("FDR 0.10", 0.10), ("FDR 0.05", 0.05)):
                s = np.array(r[m]["states"]) if q is None else fdr_select(Pm, q)
                nR = int((s == 2).sum())
                rows.append(dict(method=m, rule=rule, n=r["n"], T=r["T"],
                                 exact=float(np.all(s == t)), R_tpr=float(np.mean(s[t == 2] == 2)),
                                 R_on_fixed=float(np.mean(s[t == 1] == 2)),
                                 R_fdp=float(((s == 2) & (t != 2)).sum() / max(nR, 1))))
    d = pd.DataFrame(rows).groupby(["method", "rule", "n", "T"]).mean().reset_index()
    d.to_csv(out / "decision_rules.csv", index=False)
    designs = d[["n", "T"]].drop_duplicates().sort_values(["n", "T"]).values.tolist()
    L = []
    measures = [("R_tpr", "TPR (R)"), ("R_on_fixed", "Fixed declared R"), ("R_fdp", "FDP (R)"),
                ("exact", "Exact recovery")]
    rules = [("median", "Median"), ("FDR 0.10", "FDR, $q=0.10$"), ("FDR 0.05", "FDR, $q=0.05$")]
    for col, mlab in measures:
        for i, (rule, rlab) in enumerate(rules):
            sub = d[(d.method == "M4") & (d.rule == rule)].set_index(["n", "T"])
            head = f"\\multirow{{3}}{{*}}{{{mlab}}}" if i == 0 else ""
            vals = " & ".join(f"{sub.loc[tuple(dd), col]:.2f}" for dd in designs)
            L.append(f"{head} & {rlab} & {vals} \\\\")
        L.append("\\midrule")
    (tabdir / "dgpA_rules_rows.tex").write_text("\n".join(L[:-1]) + "\n")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/dgpA/raw")
    ap.add_argument("--raw_extra", default="results/dgpA/raw_m23,results/dgpA/raw_m6")
    ap.add_argument("--out", default="results/dgpA")
    ap.add_argument("--fig", default="../manuscript/figures")
    ap.add_argument("--tab", default="../manuscript/tables")
    a = ap.parse_args()
    global ACTIVE
    recs = load(a.raw, a.raw_extra)
    ACTIVE = [m for m in METHODS if all(m in r for r in recs)]
    print("methods:", ACTIVE)
    out, figdir, tabdir = Path(a.out), Path(a.fig), Path(a.tab)
    out.mkdir(parents=True, exist_ok=True)
    df = rep_metrics(recs)
    summ = summarize(df)
    est = estimation(recs)
    pc = per_covariate(recs)
    summ.to_csv(out / "summary_metrics.csv", index=False)
    est.to_csv(out / "estimation_metrics.csv", index=False)
    pc.to_csv(out / "per_covariate_selection.csv", index=False)
    reps = int(summ["reps"].min())
    fig_detection(summ, figdir)
    fig_heatmap(pc, figdir)
    fig_calibration(recs, figdir)
    fig_state_probs(recs, figdir)
    table_recovery(summ, tabdir, reps)
    table_estimation(est, tabdir)
    rules = table_rules(recs, tabdir, out)
    # agreement of the median probability rule with the argmax rule (Section 3.5 claim)
    for m in ("M4", "M5"):
        dis, tot = 0, 0
        for r in recs:
            Pm = np.array(r[m]["P"])
            med = np.where(Pm[:, 1] + Pm[:, 2] > 0.5, 1, 0)
            med[Pm[:, 2] > 0.5] = 2
            dis += int((med != Pm.argmax(axis=1)).sum())
            tot += Pm.shape[0]
        print(f"{m}: median vs argmax rule disagree on {dis}/{tot} candidate-replications ({dis / tot:.3%})")
    se_cols = [c for c in summ.columns if c.endswith("_se") and c != "seconds_se"]
    print("largest Monte Carlo SE in the recovery table:", float(summ[se_cols].max().max().round(4)))
    pd.set_option("display.width", 200)
    print(f"records: {len(recs)}; replications per cell: {summ.groupby(['n', 'T'])['reps'].min().to_dict()}")
    print(summ[["n", "T", "method", "exact", "acc3", "R_tpr", "R_on_fixed", "R_fdp", "incl_fpr", "seconds"]].round(3).to_string(index=False))
    print(est.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
