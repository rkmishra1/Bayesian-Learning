"""Tables and figures for the British major-road application (Section 6).

    python scripts/analyze_uk.py --fig ../manuscript/figures --tab ../manuscript/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402  (shared style)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from rpsel.application import BINARY, CONTINUOUS, LABELS, NAMES, P_CONT  # noqa: E402

BAYES = ["M4", "M5", "M6"]
GROUPS = [("Traffic (varies by year)", NAMES[:6]), ("Link geometry and network", NAMES[6:10]),
          ("Road class and country", NAMES[10:])]


def load(res: Path):
    recs = {}
    for f in sorted(res.glob("M*_c*.json")):
        r = json.loads(f.read_text())
        recs[(r["method"], r["chain"])] = r
    return recs


def fmt(x, d=2):
    return f"{x:.{d}f}".replace("-", "$-$")


# ----------------------------------------------------------------------------- descriptives
def descriptives(panel: pd.DataFrame, tabdir: Path):
    d = panel
    rows = [("Injury collisions per year", d.y), ("Fatal or serious collisions per year", d.y_ksi),
            ("Link length (km)", d.link_length_km),
            ("AADF (1{,}000 vehicles per day)", d.all_motor_vehicles / 1000),
            ("HGV share (\\%)", 100 * d.all_HGVs / d.all_motor_vehicles),
            ("LGV share (\\%)", 100 * d.LGVs / d.all_motor_vehicles),
            ("Bus and coach share (\\%)", 100 * d.buses_and_coaches / d.all_motor_vehicles),
            ("Motorcycle share (\\%)", 100 * d.two_wheeled_motor_vehicles / d.all_motor_vehicles),
            ("Pedal cycles per day", d.pedal_cycles),
            ("Curvature index", np.maximum(d.link_length_km / d.chord_km, 1.0)),
            ("Junction degree", d.node_degree),
            ("Major-road density (km/km$^2$)", d.road_density),
            ("Motorway (0/1)", d.road_category.isin(["TM", "PM"]).astype(float)),
            ("Trunk A road (0/1)", (d.road_category == "TA").astype(float)),
            ("London (0/1)", (d.region_name == "London").astype(float)),
            ("Scotland (0/1)", (d.region_name == "Scotland").astype(float)),
            ("Wales (0/1)", (d.region_name == "Wales").astype(float))]
    L = []
    num = lambda v: f"{v:,.2f}".replace(",", "{,}")
    for lab, s in rows:
        L.append(f"{lab} & {num(s.mean())} & {num(s.std())} & {num(s.min())} & {num(s.median())} & {num(s.max())} \\\\")
    (tabdir / "uk_descriptives_rows.tex").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- states
def pooled(recs, m):
    ch = [recs[k] for k in sorted(recs) if k[0] == m]
    P = np.mean([np.array(c["P"]) for c in ch], axis=0)
    out = dict(P=P, P_ri=float(np.mean([c["P_ri"] for c in ch])), chains=len(ch))
    if m == "M6":
        out["P_disp"] = np.mean([np.array(c["P_disp"]) for c in ch], axis=0)
    out["chain_gap"] = (float(np.max(np.abs(np.array(ch[0]["P"])[:, 2] - np.array(ch[1]["P"])[:, 2])))
                        if len(ch) > 1 else np.nan)
    return out


def figure_states(recs, figdir: Path):
    p = len(NAMES)
    have = [m for m in BAYES if any(k[0] == m for k in recs)]
    pool = {m: pooled(recs, m) for m in have}
    m1 = recs.get(("M1", 1))
    rows = []  # (label, key) top to bottom
    for g, names in GROUPS:
        rows.append((g, None))
        rows += [(LABELS[NAMES.index(n)], NAMES.index(n)) for n in names]
    rows.append(("Site level", None))
    rows.append(("Random intercept", "ri"))
    ypos = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 4.9), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.08))
    titles = ["(a) Included, fixed or random", "(b) Random parameter", "(c) Dispersion effect (M6)"]
    offs = dict(zip(["M1"] + have, np.linspace(0.27, -0.27, len(have) + 1)))
    for c, ax in enumerate(axes):
        for (lab, key), y0 in zip(rows, ypos):
            if key is None:
                continue
            ax.axhline(y0, color=A.GRID, lw=0.6, zorder=0)
        for m in (["M1"] if m1 else []) + have:
            sty = A.METHODS[m]
            xs, ys = [], []
            for (lab, key), y0 in zip(rows, ypos):
                if key is None:
                    continue
                if c == 2 and (m != "M6" or key == "ri"):
                    continue
                if m == "M1":
                    if c == 2:
                        continue
                    if key == "ri":
                        v = float(c == 1 or c == 0) * (1.0 if (c == 0 or m1["const_sigma"] > 0) else 0.0)
                        if c == 0:
                            continue
                    else:
                        st = m1["states"][key]
                        v = float(st >= 1) if c == 0 else float(st == 2)
                else:
                    P = pool[m]["P"]
                    if key == "ri":
                        if c == 0:
                            continue
                        v = pool[m]["P_ri"]
                    else:
                        v = (P[key, 1] + P[key, 2]) if c == 0 else (P[key, 2] if c == 1 else pool[m]["P_disp"][key])
                xs.append(v)
                ys.append(y0 + (0 if c == 2 else offs[m]))
            if not xs:
                continue
            filled = m != "M1"
            ms = {"P": 7.5, "D": 5.0}.get(sty["marker"], 5.4)
            ax.plot(xs, ys, linestyle="none", marker=sty["marker"], markersize=ms,
                    color=sty["color"] if filled else "none", markerfacecolor=sty["color"] if filled else "none",
                    markeredgecolor=sty["color"] if not filled else A.SURFACE,
                    markeredgewidth=1.3 if not filled else 0.9, zorder=3)
        ax.axvline(0.5, color=A.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.06, 1.06)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(axis="y", visible=False)
        ax.set_title(titles[c], color=A.INK, fontsize=8.5)
        A._despine(ax)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([lab for lab, _ in rows])
    for t, (lab, key) in zip(axes[0].get_yticklabels(), rows):
        if key is None:
            t.set_fontweight("semibold")
            t.set_color(A.INK)
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    fig.supxlabel("Posterior probability (Bayesian methods) or selection (M1, open diamond)", fontsize=8,
                  color=A.INK2, y=0.01)
    handles = [Line2D([], [], linestyle="none", marker="D", markersize=5, markerfacecolor="none",
                      markeredgecolor=A.METHODS["M1"]["color"], markeredgewidth=1.3, label=A.METHODS["M1"]["label"])]
    handles += [Line2D([], [], linestyle="none", marker=A.METHODS[m]["marker"],
                       markersize={"P": 7.5}.get(A.METHODS[m]["marker"], 5.4), color=A.METHODS[m]["color"],
                       markeredgecolor=A.SURFACE, markeredgewidth=0.9, label=A.METHODS[m]["label"]) for m in have]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.905),
               handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    A._save(fig, figdir, "uk_state_probabilities")


# ----------------------------------------------------------------------------- estimates
def trace_summaries(res: Path, m: str):
    tr = [np.load(f) for f in sorted(res.glob(f"{m}_c*_trace.npz"))]
    cat = lambda k: np.concatenate([t[k] for t in tr])
    beta, sigma, gam, dlt = cat("beta"), cat("sigma"), cat("gamma"), cat("delta")
    out = dict(beta=beta, sig=np.abs(sigma), gam=gam, dlt=dlt)
    if m == "M6":
        out.update(rho=cat("rho"), eta=cat("eta"), rho0=cat("rho0"))
    else:
        out["r"] = cat("r")
    return out


def table_estimates(recs, res: Path, tabdir: Path):
    """M6 (model-averaged) next to M1, for every candidate and the intercept heterogeneity."""
    t = trace_summaries(res, "M6")
    m1 = recs[("M1", 1)]
    p = len(NAMES)
    L = []

    def ci(x):
        lo, hi = np.quantile(x, [0.025, 0.975])
        return f"{fmt(x.mean())} [{fmt(lo)}, {fmt(hi)}]"

    for g, names in GROUPS:
        L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{g}}}}} \\\\")
        for n in names:
            j = NAMES.index(n)
            b = ci(t["beta"][:, j])
            pr = t["dlt"][:, j].mean()
            s = ci(t["sig"][:, j][t["dlt"][:, j]]) if pr >= 0.05 else "--"
            pd_ = t["eta"][:, j].mean()
            rho = ci(t["rho"][:, j][t["eta"][:, j]]) if pd_ >= 0.05 else "--"
            st = m1["states"][j]
            if st == 0:
                m1b, m1s = "--", "--"
            else:
                m1b = f"{fmt(m1['beta'][j])} ({fmt(m1['beta_se'][j])})"
                m1s = f"{fmt(m1['sigma'][j])} ({fmt(m1['sigma_se'][j])})" if st == 2 else "--"
            L.append(f"\\quad {LABELS[j]} & {b} & {fmt(pr)} & {s} & {fmt(pd_)} & {m1b} & {m1s} \\\\")
    pr0 = t["dlt"][:, p].mean()
    s0 = ci(t["sig"][:, p][t["dlt"][:, p]]) if pr0 >= 0.05 else "--"
    m1s0 = f"{fmt(m1['const_sigma'])} ({fmt(m1['const_sigma_se'])})" if m1["const_sigma"] > 0 else "--"
    L.append("\\addlinespace")
    L.append(f"Random intercept & -- & {fmt(pr0)} & {s0} & -- & -- & {m1s0} \\\\")
    (tabdir / "uk_estimates_rows.tex").write_text("\n".join(L) + "\n")


def squares_selected(res: Path, m: str):
    t = trace_summaries(res, m)
    p = len(NAMES)
    inc = t["gam"][:, p:p + P_CONT].mean(axis=0)
    return {NAMES[k]: float(inc[k]) for k in range(P_CONT)}


# ----------------------------------------------------------------------------- prediction
def table_predict(recs, res: Path, tabdir: Path):
    have = [m for m in ["M1"] + BAYES if any(k[0] == m for k in recs)]
    lpd = {m: np.mean([np.load(f) for f in sorted(res.glob(f"{m}_c*_lpd.npy"))], axis=0) for m in have}
    best = max(have, key=lambda m: lpd[m].sum())
    rows, L = [], []
    for m in have:
        ch = [recs[k] for k in sorted(recs) if k[0] == m]
        d = lpd[m] - lpd[best]
        se = np.sqrt(d.size) * d.std(ddof=1)
        n_rand = (sum(s == 2 for s in ch[0]["states"][:len(NAMES)]) if m != "M1"
                  else sum(s == 2 for s in ch[0]["states"]))
        ri = (np.mean([c["P_ri"] for c in ch]) > 0.5) if m != "M1" else ch[0]["const_sigma"] > 0
        mae = np.mean([c["predict"]["mae"] for c in ch])
        rmse = np.mean([c["predict"]["rmse"] for c in ch])
        sec = np.mean([c["seconds"] for c in ch]) / 60
        row = dict(method=m, elpd=lpd[m].sum(), d_elpd=d.sum(), se=se, mae=mae, rmse=rmse, n_random=n_rand,
                   ri=bool(ri), minutes=sec)
        rows.append(row)
        dtxt = "0" if m == best else f"{fmt(d.sum(), 1)} ({se:.1f})"
        L.append(f"{A.METHODS[m]['label']} & {n_rand}{' + RI' if ri else ''} & {lpd[m].sum():,.1f} & {dtxt} & "
                 f"{mae:.3f} & {rmse:.3f} & {sec:.0f} \\\\".replace(",", "{,}").replace("-", "$-$"))
    (tabdir / "uk_predict_rows.tex").write_text("\n".join(L) + "\n")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/uk")
    ap.add_argument("--data", default="data/uk_links_2015_2019.csv.gz")
    ap.add_argument("--fig", default="../manuscript/figures")
    ap.add_argument("--tab", default="../manuscript/tables")
    a = ap.parse_args()
    res, figdir, tabdir = Path(a.res), Path(a.fig), Path(a.tab)
    tabdir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(a.data)
    descriptives(panel, tabdir)
    recs = load(res)
    if not recs:
        print("no application results yet")
        return
    for m in BAYES:
        if any(k[0] == m for k in recs):
            po = pooled(recs, m)
            print(m, "chains", po["chains"], "max |dP(R)| between chains", round(po["chain_gap"], 3),
                  "P(RI)", round(po["P_ri"], 3))
            print("   P(R):", dict(zip(NAMES, po["P"][:, 2].round(2))))
            print("   P(incl):", dict(zip(NAMES, (po["P"][:, 1] + po["P"][:, 2]).round(2))))
            if m == "M6":
                print("   P(disp):", dict(zip(NAMES, po["P_disp"].round(2))))
            if m in ("M5", "M6"):
                print("   squares:", {k: round(v, 2) for k, v in squares_selected(res, m).items()})
    if ("M1", 1) in recs:
        m1 = recs[("M1", 1)]
        print("M1 states", dict(zip(NAMES, m1["states"])), "const sigma", round(m1["const_sigma"], 3))
    figure_states(recs, figdir)
    if ("M1", 1) in recs and any(k[0] == "M6" for k in recs):
        table_estimates(recs, res, tabdir)
    pr = table_predict(recs, res, tabdir)
    print(pr.round(3).to_string(index=False))
    pr.to_csv(res / "predict_summary.csv", index=False)


if __name__ == "__main__":
    main()
