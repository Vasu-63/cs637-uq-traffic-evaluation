"""
Slide figures for the calibration, sensor-failure and latency experiments,
built only from the CSVs in results/ (so every plotted number traces to a file).

CS637 project. Run from this folder:  python make_figures.py
Writes figures/*.png (and prints the numbers each figure shows).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# one fixed colour per model family member, used in every figure
COLORS = {"plain": "#2a78d6", "gpde": "#eb6834", "phy": "#1baf7a"}
LABEL = {"plain": "Plain", "gpde": "+ GPDE", "phy": "+ GPDE + Phy"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
QUANTILE = {"quantile_model_PEMSD4": "plain", "gpde_quantile_model_PEMSD4": "gpde",
            "gpde_quantile_phy_model_PEMSD4": "phy"}
MAEMIS = {"maemis_model_PEMSD4": "plain", "gpde_maemis_model_PEMSD4": "gpde",
          "gpde_maemis_phy_model_PEMSD4_0.005": "phy"}
MINUTES = np.arange(5, 65, 5)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 200,
})


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=10, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.plot([], [], color=color)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- coverage
def fig_coverage():
    cov = pd.read_csv(os.path.join(RES, "coverage_conformal_metrics.csv"))
    cov = cov[cov.model.isin(QUANTILE)].copy()
    cov["m"] = cov.model.map(QUANTILE)
    per_h = cov[cov.horizon != "avg_1-12"].copy()
    per_h["min"] = per_h.horizon.str.replace("min", "").astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.axhline(0.95, color=MUTED, lw=1, ls="--")
    ax.text(60, 0.958, "target 95%", color=MUTED, fontsize=10, ha="right")
    for m in ("plain", "gpde", "phy"):
        r = per_h[(per_h.m == m) & (per_h.method == "raw")].sort_values("min")
        ax.plot(r["min"], r.coverage, color=COLORS[m], lw=2, marker="o", ms=5)
        end_label(ax, 60, r.coverage.iloc[-1], f"{LABEL[m]}  {r.coverage.iloc[-1]:.0%}", COLORS[m],
                  dy={"plain": 0, "gpde": -9, "phy": 4}[m])
        a = per_h[(per_h.m == m) & (per_h.method == "aci")].sort_values("min")
        ax.plot(a["min"], a.coverage, color=COLORS[m], lw=1.5, ls=":", alpha=0.9)
    ax.text(60, 0.925, "calibrated (ACI, dotted): 95% at every horizon",
            color=MUTED, fontsize=10, ha="right")
    ax.set(title="Coverage of the nominal 95% interval", xlabel="Forecast horizon (min)",
           ylabel="Share of true speeds inside the interval", ylim=(0.45, 1.0), xticks=MINUTES)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    ax = axes[1]
    for m in ("plain", "gpde", "phy"):
        r = per_h[(per_h.m == m) & (per_h.method == "raw")].sort_values("min")
        a = per_h[(per_h.m == m) & (per_h.method == "aci")].sort_values("min")
        ax.plot(r["min"], r.width, color=COLORS[m], lw=2, marker="o", ms=5)
        ax.plot(a["min"], a.width, color=COLORS[m], lw=2, ls=":", marker="o", ms=4,
                mfc="white")
        end_label(ax, 60, a.width.iloc[-1], f"{LABEL[m]}  {a.width.iloc[-1]:.0f} mph", COLORS[m],
                  dy={"plain": 0, "gpde": 5, "phy": -5}[m])
    ax.text(60, 1.6, "authors' intervals (solid): narrower as the horizon grows",
            color=MUTED, fontsize=10, ha="right")
    ax.text(18, 22, "calibrated (dotted)", color=MUTED, fontsize=10)
    ax.set(title="Interval width needed for honest 95% coverage", xlabel="Forecast horizon (min)",
           ylabel="Mean interval width (mph)", ylim=(0, 31), xticks=MINUTES)
    fig.tight_layout(w_pad=4)
    save(fig, "coverage_by_horizon.png")


def fig_interval_score():
    """Raw vs calibrated interval score per model, both UQ methods."""
    own = pd.read_csv(os.path.join(RES, "coverage_conformal_metrics.csv"))
    sh = pd.read_csv(os.path.join(RES, "coverage_conformal_metrics_shared_predictions.csv"))
    rows = []
    for fam, mapping, src in (("Quantile", QUANTILE, own), ("MAEMIS", MAEMIS, sh)):
        d = src[src.model.isin(mapping) & (src.horizon == "avg_1-12")]
        for _, r in d.iterrows():
            if r.method in ("raw", "aci"):
                rows.append({"family": fam, "m": mapping[r.model], "method": r.method,
                             "MIS": r.MIS, "coverage": r.coverage})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, fam in zip(axes, ("Quantile", "MAEMIS")):
        x = np.arange(3)
        for i, m in enumerate(("plain", "gpde", "phy")):
            raw = df[(df.family == fam) & (df.m == m) & (df.method == "raw")].iloc[0]
            cal = df[(df.family == fam) & (df.m == m) & (df.method == "aci")].iloc[0]
            ax.bar(i - 0.19, raw.MIS, 0.36, color=COLORS[m], alpha=0.35, edgecolor="white", lw=2)
            ax.bar(i + 0.19, cal.MIS, 0.36, color=COLORS[m], edgecolor="white", lw=2)
            ax.text(i - 0.19, raw.MIS + 0.8, f"{raw.MIS:.1f}\n({raw.coverage:.0%})", ha="center",
                    va="bottom", fontsize=9, color=INK)
            ax.text(i + 0.19, cal.MIS + 0.8, f"{cal.MIS:.1f}\n(95%)", ha="center", va="bottom",
                    fontsize=9, color=INK)
        ax.set_xticks(x, [LABEL[m] for m in ("plain", "gpde", "phy")])
        ax.set_title(f"{fam} models, PEMSD4, all 12 horizons")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Interval score (lower is better)")
    axes[0].set_ylim(0, 72)
    axes[0].text(-0.45, 68, "light: authors' intervals (coverage)    dark: after calibration (ACI)",
                 fontsize=10, color=MUTED)
    fig.tight_layout(w_pad=3)
    save(fig, "interval_score_raw_vs_calibrated.png")
    print(df.round(3).to_string())


# ---------------------------------------------------------- sensor failure
def fig_sensor_failure():
    sf = pd.read_csv(os.path.join(RES, "sensor_failure_metrics.csv"))
    fx = pd.read_csv(os.path.join(RES, "sensor_failure_dead_only_3seeds_metrics.csv"))
    sf, fx = (d[(d.horizon == "avg_1-12") & d.model.isin(QUANTILE)].copy() for d in (sf, fx))
    for d in (sf, fx):
        d["m"] = d.model.map(QUANTILE)
    clean = sf[(sf["mode"] == "clean") & (sf.group == "all")].set_index("m").MAE

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for m in ("plain", "gpde", "phy"):
        r = sf[(sf["mode"] == "dead") & (~sf.fix) & (sf.group == "healthy") & (sf.m == m)]
        r = r.groupby("fraction").MAE.mean()
        xs = [0] + list(r.index * 100)
        ys = [clean[m]] + list(r.values)
        ax.plot(xs, ys, color=COLORS[m], lw=2, marker="o", ms=5)
        end_label(ax, 40, ys[-1], f"{LABEL[m]}  {ys[-1]:.1f}", COLORS[m],
                  dy={"plain": 0, "gpde": 7, "phy": -6}[m])
    ax.set(title="Dead sensors leak into healthy ones (no fix)",
           xlabel="Sensors dead (%)", ylabel="MAE on the healthy sensors (mph)",
           ylim=(0, 7.5), xticks=[0, 5, 10, 20, 40])

    ax = axes[1]
    for m in ("plain", "gpde", "phy"):
        r = sf[(sf["mode"] == "dead") & (~sf.fix) & (sf.group == "all") & (sf.m == m)]
        r = r.groupby("fraction").MAE.mean()
        f = fx[(fx["mode"] == "dead") & (fx.fix) & (fx.group == "all") & (fx.m == m)]
        f = f.groupby("fraction").MAE.mean()
        ax.plot([0] + list(r.index * 100), [clean[m]] + list(r.values), color=COLORS[m], lw=2,
                marker="o", ms=5)
        ax.plot([0] + list(f.index * 100), [clean[m]] + list(f.values), color=COLORS[m], lw=2,
                ls=":", marker="o", ms=4, mfc="white")
        end_label(ax, 40, r.values[-1], f"{LABEL[m]}  {r.values[-1]:.1f}", COLORS[m],
                  dy={"plain": 0, "gpde": -2, "phy": 7}[m])
    ax.text(41, 3.3, "with fix\n3.1 to 3.5", fontsize=10, color=MUTED, va="center")
    ax.text(12, 11.5, "no fix (solid)", fontsize=10, color=MUTED)
    ax.text(12, 1.0, "detect dead sensor, impute from graph neighbours (dotted)",
            fontsize=10, color=MUTED)
    ax.set(title="Overall error, and a simple fix", xlabel="Sensors dead (%)",
           ylabel="MAE on all sensors (mph)", ylim=(0, 17), xlim=(-1, 50),
           xticks=[0, 5, 10, 20, 40])
    fig.tight_layout(w_pad=6)
    save(fig, "dead_sensors.png")


# ----------------------------------------------------------------- latency
def fig_latency():
    lat = pd.read_csv(os.path.join(RES, "latency.csv"))
    size = pd.read_csv(os.path.join(RES, "model_size.csv")).set_index("model")
    lat = lat[lat.batch == 1].set_index("model")
    order = list(QUANTILE)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    y = np.arange(3)[::-1]
    for yi, mod in zip(y, order):
        m = QUANTILE[mod]
        ax.barh(yi, lat.loc[mod, "median_ms"], 0.55, color=COLORS[m])
        ax.text(lat.loc[mod, "median_ms"] + 1.2, yi,
                f"{lat.loc[mod, 'median_ms']:.0f} ms   ({int(size.loc[mod, 'params']):,} params, "
                f"{size.loc[mod, 'checkpoint_MB']:.2f} MB)", va="center", fontsize=10)
    ax.set_yticks(y, [LABEL[QUANTILE[mod]] for mod in order])
    ax.set(title="One network-wide forecast on 1 CPU core (batch 1, median)",
           xlabel="Latency (ms); budget = 5-min sampling period = 300,000 ms", xlim=(0, 115))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    fig.tight_layout()
    save(fig, "latency.png")


if __name__ == "__main__":
    fig_coverage()
    fig_interval_score()
    fig_sensor_failure()
    fig_latency()
