"""
Robustness to realistic sensor faults, and a cheap mitigation.

CS637 project. The paper's threat model is gradient attacks only
(FGSM, 2-step PGD; critique point 10). Loop detectors in the field fail in
duller ways. We break a growing fraction of sensors for the whole test period
and see what happens to the forecast and to its 95% interval.

Fault modes (applied to the 12-step model input, all three channels; the
ground truth is untouched because we score against what traffic really did):
  dead   the sensor reports 0 (how PeMS stores missing readings)
  stuck  the sensor repeats the reading from the first input step for the whole
         hour-long window (frozen value)

Mitigation ("+fix"), a pre-processing step any deployment could run:
  detect a sensor as faulty in a window if it is all zeros or flat
  (zero variance over the 12 steps in every channel), then replace its inputs
  with the weighted mean of its non-faulty graph neighbours, falling back to
  the sensor's training mean. The detector's false-alarm rate on clean data is
  reported.

Scores: MAE, RMSE, coverage, width, MIS on (a) all sensors, (b) the failed
sensors, (c) healthy sensors that are direct graph neighbours of a failed one
(does the fault spread through the diffusion convolution?).

Fractions of failed sensors are nested (the 10% set contains the 5% set) and
drawn with `--seeds` different random sensor sets.

Usage (Colab, GPU recommended):
    python sensor_failure.py \
        --dataset_dir /content/data/PEMSD4 \
        --variant /content/drive/MyDrive/CS637/extensions_out/models/quantile_model_PEMSD4 \
        --variant .../gpde_quantile_model_PEMSD4 \
        --variant .../gpde_quantile_phy_model_PEMSD4 \
        --out /content/drive/MyDrive/CS637/extensions_out/sensor_failure
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_utils import load_variant, predict          # noqa: E402
from coverage_conformal import interval_metrics            # noqa: E402

HERE = os.getcwd()
ALPHA = 0.05


# ------------------------------------------------------------------ faults
def inject(x, failed, mode):
    """x: (N, 12, S, D) raw units. failed: bool (S,). Returns a copy."""
    x = x.copy()
    if mode == "dead":
        x[:, :, failed, :] = 0.0
    elif mode == "stuck":
        x[:, :, failed, :] = x[:, :1, failed, :]
    else:
        raise ValueError(mode)
    return x


def detect(x, rule="zero_or_flat"):
    """Flag (N, S). rule "zero_or_flat": all-zero or flat over the window in
    every channel (original rule). rule "zero": all-zero only, so a stuck
    sensor keeps its own stale reading (first run showed that replacing it
    by the neighbours' mean is worse than leaving it)."""
    allzero = np.all(x == 0, axis=(1, 3))
    if rule == "zero":
        return allzero
    flat = np.all(x.max(axis=1) == x.min(axis=1), axis=-1)
    return allzero | flat


def impute(x, flags, W, train_mean):
    """Replace flagged sensor inputs by the weighted mean of unflagged
    neighbours (symmetrised adjacency W, no self loops); sensors with no
    healthy neighbour get their training mean per channel."""
    x = x.copy()
    N = x.shape[0]
    for i in range(N):
        f = flags[i]
        if not f.any():
            continue
        w = W[f][:, ~f]                                   # (n_failed, n_ok)
        tot = w.sum(1)
        ok = tot > 0
        src = x[i][:, ~f, :]                              # (12, n_ok, D)
        est = np.einsum("fk,tkd->tfd", w[ok] / tot[ok, None], src)
        idx = np.where(f)[0]
        x[i][:, idx[ok], :] = est
        if (~ok).any():
            x[i][:, idx[~ok], :] = train_mean[idx[~ok]][None]
    return x


# ------------------------------------------------------------------- scoring
def score(pred_mph, y, groups):
    rows = {}
    for gname, sel in groups.items():
        if sel is None or not sel.any():
            continue
        lo, md, hi = (pred_mph[..., k][:, :, sel] for k in range(3))
        yy = y[:, :, sel]
        r_all = interval_metrics(lo, md, hi, yy, ALPHA)
        rows[(gname, "avg_1-12")] = r_all
        for tag, h in (("15min", 2), ("30min", 5), ("60min", 11)):
            rows[(gname, tag)] = interval_metrics(lo[h], md[h], hi[h], yy[h], ALPHA)
    return rows


KEEP = ["n", "MAE", "RMSE", "coverage", "width", "MIS", "MIS_authors",
        "cov[congested <40 mph]"]


def main(a):
    os.makedirs(a.out, exist_ok=True)
    d = os.path.abspath(os.path.join(HERE, a.dataset_dir))
    print("loading data from", d)
    tr = np.load(os.path.join(d, "train.npz"))["x"]
    mean, std = float(tr[..., 0].mean()), float(tr[..., 0].std())   # the authors' scaler
    train_mean = tr.reshape(-1, tr.shape[2], tr.shape[3]).mean(0)    # (S, D)
    del tr
    te = np.load(os.path.join(d, "test.npz"))
    x_raw = te["x"].astype(np.float32)                    # (N, 12, S, D)
    y = te["y"][..., 0].transpose(1, 0, 2).astype(np.float64)   # (12, N, S) mph
    y[y == 0] = np.nan
    if a.max_samples:
        x_raw, y = x_raw[: a.max_samples], y[:, : a.max_samples]
    N, T, S, D = x_raw.shape
    print(f"test N={N} sensors={S} features={D}; scaler mean {mean:.3f} std {std:.3f}")

    def run_model(model, x):
        xs = x.copy()
        xs[..., 0] = (xs[..., 0] - mean) / std
        p = predict(model, xs, a.batch_size)
        return p * std + mean                             # (12, N, S, 3) mph

    clean_flags = detect(x_raw, a.detect)
    false_alarm = float(clean_flags.mean())
    print(f"detector rule '{a.detect}': flags {false_alarm:.4%} of clean test "
          f"sensor-windows (genuinely missing or flat readings)")
    modes = [m.strip() for m in a.modes.split(",")]

    fractions = [float(f) for f in a.fractions.split(",")]
    rows = []
    t_start = time.time()
    for vdir in a.variant:
        vdir = os.path.abspath(os.path.join(HERE, vdir))
        name = os.path.basename(vdir.rstrip("/"))
        model, cfg, info = load_variant(vdir)
        os.chdir(HERE)
        W = info["adj_mx"].astype(np.float64)
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0)
        print(f"\n== {name} (epoch {info['epoch']}, {info['device']})")

        p = run_model(model, x_raw)
        base = score(p, y, {"all": np.ones(S, bool)})
        print(f"  clean: MAE {base[('all', 'avg_1-12')]['MAE']:.4f}  "
              f"coverage {base[('all', 'avg_1-12')]['coverage']:.4f}  "
              "(compare with Table 2 / training log; our number has no padded samples)")
        for (g, h), r in base.items():
            rows.append({"model": name, "mode": "clean", "fix": False, "fraction": 0.0,
                         "seed": -1, "group": g, "horizon": h,
                         **{k: r[k] for k in KEEP}})
        if a.clean_only:
            continue

        for seed in range(a.seeds):
            order = np.random.default_rng(1000 + seed).permutation(S)
            for frac in fractions:
                k = max(1, int(round(frac * S)))
                failed = np.zeros(S, bool)
                failed[order[:k]] = True
                nbr = (W[failed].sum(0) > 0) & ~failed
                groups = {"all": np.ones(S, bool), "failed": failed,
                          "healthy_neighbours": nbr, "healthy": ~failed}
                for mode in modes:
                    xf = inject(x_raw, failed, mode)
                    variants = [] if a.fix_only else [(False, xf)]
                    if not a.no_fix:
                        flags = detect(xf, a.detect)
                        variants.append((True, impute(xf, flags, W, train_mean)))
                        recall = flags[:, failed].mean()
                    for fix, xin in variants:
                        res = score(run_model(model, xin), y, groups)
                        for (g, h), r in res.items():
                            rows.append({"model": name, "mode": mode, "fix": fix,
                                         "fraction": frac, "seed": seed, "group": g,
                                         "horizon": h, **{kk: r[kk] for kk in KEEP}})
                        r = res[("all", "avg_1-12")]
                        rf = res[("failed", "avg_1-12")]
                        print(f"  seed {seed} {mode:<5} {frac:>4.0%} fix={int(fix)}: "
                              f"all MAE {r['MAE']:.3f} cov {r['coverage']:.3f} | "
                              f"failed MAE {rf['MAE']:.3f} cov {rf['coverage']:.3f} width {rf['width']:.2f}"
                              + (f" | detector recall {recall:.3f}" if fix else ""))
        del model

    path = os.path.join(a.out, "sensor_failure_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(a.out, "sensor_failure_meta.json"), "w") as f:
        json.dump({"dataset_dir": d, "test_samples": N, "sensors": S,
                   "fractions": fractions, "seeds": a.seeds, "alpha": ALPHA,
                   "modes": modes, "detector_rule": a.detect, "fix_only": a.fix_only,
                   "detector_false_alarm_rate_clean": false_alarm,
                   "minutes": (time.time() - t_start) / 60}, f, indent=1)
    print("\nwrote", path)
    if not a.clean_only:
        plot(rows, a.out)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[(df.horizon == "avg_1-12")]
    models = list(dict.fromkeys(df.model))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e4e4", "grid.linewidth": 0.6})
    for metric, ylabel in (("MAE", "MAE (mph), all sensors"),
                           ("coverage", "Coverage of 95% interval, failed sensors")):
        group = "all" if metric == "MAE" else "failed"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, mode in zip(axes, ("dead", "stuck")):
            for c, m in zip(colors, models):
                clean = df[(df.model == m) & (df["mode"] == "clean") & (df.group == "all")][metric]
                for fix, ls in ((False, "-"), (True, "--")):
                    s = df[(df.model == m) & (df["mode"] == mode) & (df.fix == fix)
                           & (df.group == group)].groupby("fraction")[metric].mean()
                    if s.empty:
                        continue
                    xs = [0.0] + list(s.index)
                    ys = [float(clean.iloc[0])] + list(s.values) if metric == "MAE" \
                        else [np.nan] + list(s.values)
                    ax.plot(np.array(xs) * 100, ys, ls, marker="o", ms=5, lw=2, color=c,
                            label=f"{m}{' + fix' if fix else ''}")
            if metric == "coverage":
                ax.axhline(0.95, color="#555", lw=1, ls=":")
            ax.set(title=f"{mode} sensors", xlabel="Failed sensors (%)")
        axes[0].set_ylabel(ylabel)
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        f = os.path.join(out, f"sensor_failure_{metric}.png")
        fig.savefig(f, dpi=160)
        plt.close(fig)
        print("wrote", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True, help="folder with train/val/test.npz (PEMSD4)")
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", default="sensor_failure_out")
    ap.add_argument("--fractions", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=0, help="debug: first N test samples only")
    ap.add_argument("--no_fix", action="store_true")
    ap.add_argument("--fix_only", action="store_true",
                    help="skip the no-fix runs (they do not depend on the detector rule)")
    ap.add_argument("--detect", choices=["zero_or_flat", "zero"], default="zero_or_flat",
                    help="which sensor-windows the fix replaces")
    ap.add_argument("--modes", default="dead,stuck", help="fault modes to run")
    ap.add_argument("--clean_only", action="store_true", help="only check the clean score")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    main(a)
