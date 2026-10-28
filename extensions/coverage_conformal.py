"""
Coverage metrics and conformal calibration of the 95% prediction intervals.

CS637 project (Bao et al., TCPS 2026).

What it does
------------
1. Measures how well the raw quantile intervals [q0.025, q0.975] are calibrated:
   coverage (PICP), mean width, the standard Mean Interval Score, the authors'
   "MIS" (standard MIS + MAE, see finding 13), quantile-crossing rate,
   per-sensor coverage spread, and coverage split by traffic regime
   (congested / transitional / free flow).
2. Recalibrates the intervals three ways, using the validation split as the
   calibration set and the test split for evaluation:
     cqr        split conformalized quantile regression (Romano et al. 2019),
                one correction per forecast horizon
     cqr_sensor the same, one correction per (horizon, sensor)  ("Mondrian" CQR)
     aci        adaptive conformal inference (Gibbs & Candes 2021) run online
                over the test period, per (horizon, sensor), with the feedback
                delayed by the horizon (the truth for an h-step forecast is only
                known h steps later)
3. Writes one CSV of metrics and two figures per run.

Input
-----
the team's prediction files (README in outputs/pemsd4/), or make_predictions.py output:
    val_pred / test_pred   (12, N, S, 3)  lower / median / upper, mph
    val_truth / test_truth (12, N, S)     mph
Zeros in the truth are missing readings and are masked, as in the authors' code.

Usage
-----
    python coverage_conformal.py \
        --npz quantile=/content/drive/MyDrive/CS637/outputs/pemsd4/quantile.npz \
        --npz gpde_quantile=.../gpde_quantile.npz \
        --npz gpde_quantile_phy=.../gpde_quantile_phy.npz \
        --out /content/drive/MyDrive/CS637/extensions_out/coverage

    python coverage_conformal.py --selftest      # synthetic data, no files needed
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ALPHA_DEFAULT = 0.05
REGIMES = [("congested <40 mph", 0.0, 40.0),
           ("transitional 40-60", 40.0, 60.0),
           ("free flow >=60", 60.0, np.inf)]
REPORT_HORIZONS = {"15min": 2, "30min": 5, "60min": 11}   # index into 12 steps of 5 min


# --------------------------------------------------------------------------- io
def drop_padding(pred, truth, name, split):
    """The authors' DataLoader pads the last batch with copies of the final
    sample. Consecutive sliding windows are never identical, so trailing
    duplicates are padding. Keep one copy."""
    n = truth.shape[1]
    k = 0
    last = truth[:, -1]
    while k + 1 < n and np.array_equal(truth[:, n - 2 - k], last):
        k += 1
    if k:
        print(f"  [{name}/{split}] dropping {k} padded duplicate samples "
              f"({n} -> {n - k})")
        pred, truth = pred[:, : n - k], truth[:, : n - k]
    return pred, truth


def load_npz(path, name):
    d = np.load(path, allow_pickle=True)
    out = {}
    for split in ("val", "test"):
        p = d[f"{split}_pred"].astype(np.float64)
        t = d[f"{split}_truth"].astype(np.float64)
        if p.ndim != 4 or p.shape[-1] != 3 or p.shape[:3] != t.shape:
            raise ValueError(f"{path}: {split}_pred {p.shape} / {split}_truth "
                             f"{t.shape} not (12,N,S,3)/(12,N,S)")
        out[split] = drop_padding(p, t, name, split)
    print(f"  [{name}] val N={out['val'][1].shape[1]}  "
          f"test N={out['test'][1].shape[1]}  sensors={out['test'][1].shape[2]}")
    return out


# ---------------------------------------------------------------------- metrics
def interval_metrics(lo, med, hi, y, alpha, sensor_axis=2):
    """All arrays (..., N, S) with NaN in y where the reading is missing.
    Returns a dict of scalars over every non-missing entry."""
    m = np.isfinite(y)
    yy, l, md, u = y[m], lo[m], med[m], hi[m]
    inside = (yy >= l) & (yy <= u)
    width = u - l
    below = np.clip(l - yy, 0, None)
    above = np.clip(yy - u, 0, None)
    mis = width + (2 / alpha) * (below + above)      # standard interval score
    abs_err = np.abs(yy - md)
    res = {
        "n": int(m.sum()),
        "MAE": abs_err.mean(),
        "RMSE": math.sqrt(((yy - md) ** 2).mean()),
        "coverage": inside.mean(),
        "width": width.mean(),
        "MIS": mis.mean(),
        "MIS_authors": (mis + abs_err).mean(),   # what the authors' mis_loss reports
        "crossing_rate": ((l > md) | (md > u)).mean(),
    }
    # coverage by regime of the true speed
    for label, a, b in REGIMES:
        sel = (yy >= a) & (yy < b)
        res[f"cov[{label}]"] = inside[sel].mean() if sel.any() else np.nan
        res[f"share[{label}]"] = sel.mean()
    # per-sensor coverage spread
    ins_full = np.where(m, (y >= lo) & (y <= hi), np.nan)
    axes = tuple(i for i in range(y.ndim) if i != y.ndim - 1)
    per_sensor = np.nanmean(ins_full, axis=axes)
    res["sensor_cov_min"] = np.nanmin(per_sensor)
    res["sensor_cov_p05"] = np.nanpercentile(per_sensor, 5)
    res["sensor_cov_median"] = np.nanmedian(per_sensor)
    res["frac_sensors_below_90"] = np.nanmean(per_sensor < 0.90)
    return res


def per_sensor_coverage(lo, hi, y):
    m = np.isfinite(y)
    ins = np.where(m, (y >= lo) & (y <= hi), np.nan)
    return np.nanmean(ins, axis=tuple(range(y.ndim - 1)))


# --------------------------------------------------------------------- conformal
def cqr_scores(lo, hi, y):
    """Conformity score of CQR: positive when y falls outside [lo, hi]."""
    return np.maximum(lo - y, y - hi)          # NaN where y is NaN


def conformal_quantile(scores, alpha, axis):
    """Finite-sample corrected quantile: the ceil((n+1)(1-alpha))-th smallest
    score along `axis`, ignoring NaN. Returns +inf when n is too small."""
    s = np.sort(scores, axis=axis)             # NaN sorted to the end
    n = np.sum(np.isfinite(scores), axis=axis)
    k = np.ceil((n + 1) * (1 - alpha)).astype(int) - 1     # 0-based rank
    too_small = k > n - 1
    k = np.clip(k, 0, None)
    k = np.minimum(k, np.maximum(n - 1, 0))
    q = np.take_along_axis(s, np.expand_dims(k, axis), axis=axis).squeeze(axis)
    return np.where(too_small, np.inf, q)


def apply_correction(lo, med, hi, q):
    lo2, hi2 = lo - q, hi + q
    # a negative correction can make the interval empty; collapse it to the median
    bad = lo2 > hi2
    lo2 = np.where(bad, med, lo2)
    hi2 = np.where(bad, med, hi2)
    return lo2, hi2


def cqr(val, test, alpha, per_sensor):
    (vp, vy), (tp, ty) = val, test
    s = cqr_scores(vp[..., 0], vp[..., 2], vy)              # (12, Nv, S)
    if per_sensor:
        q = conformal_quantile(s, alpha, axis=1)[:, None, :]       # (12, 1, S)
    else:
        flat = s.reshape(s.shape[0], -1)
        q = conformal_quantile(flat, alpha, axis=1)[:, None, None]  # (12, 1, 1)
    lo, hi = apply_correction(tp[..., 0], tp[..., 1], tp[..., 2], q)
    return lo, hi, q


def aci(val, test, alpha, gamma=0.005):
    """Adaptive conformal inference over the test period, one alpha_t per
    (horizon, sensor). Calibration scores come from the validation split; the
    level alpha_t moves after every miss/hit, and the hit/miss of an h-step
    forecast issued at window t is only used at window t+h."""
    (vp, vy), (tp, ty) = val, test
    H, N, S = ty.shape
    s = np.sort(cqr_scores(vp[..., 0], vp[..., 2], vy), axis=1)   # (12, Nv, S)
    n = np.sum(np.isfinite(s), axis=1)                             # (12, S)
    smax = np.nanmax(s, axis=1)
    lo_t, med_t, hi_t = tp[..., 0], tp[..., 1], tp[..., 2]
    lo_out = np.empty_like(lo_t)
    hi_out = np.empty_like(hi_t)
    a_t = np.full((H, S), alpha)
    err = np.full((H, N, S), np.nan)
    delay = np.arange(1, H + 1)                                    # steps until truth
    hh = np.arange(H)[:, None]
    ss = np.arange(S)[None, :]
    alpha_trace = np.empty((N, H))
    capped = np.zeros((H, S))
    for t in range(N):
        # update with forecasts whose truth has just become available
        src = t - delay                                            # (H,)
        ok = src >= 0
        if ok.any():
            e = err[hh[ok, 0], src[ok], :]                         # (h_ok, S)
            upd = np.where(np.isfinite(e), alpha - e, 0.0)
            a_t[ok] = a_t[ok] + gamma * upd
        level = 1 - a_t
        k = np.ceil((n + 1) * level).astype(int) - 1
        inf_mask = (a_t <= 0) | (k > n - 1)
        k = np.clip(k, 0, np.maximum(n - 1, 0))
        q = s[hh, k, ss]                                           # (H, S)
        # textbook ACI uses an infinite interval here; we cap at the largest
        # calibration score so widths stay interpretable, and count the caps
        q = np.where(inf_mask, smax, q)
        capped += inf_mask
        q = np.where(a_t >= 1, -np.inf, q)
        lo, hi = apply_correction(lo_t[:, t], med_t[:, t], hi_t[:, t], q)
        lo_out[:, t], hi_out[:, t] = lo, hi
        y = ty[:, t]
        err[:, t] = np.where(np.isfinite(y), ((y < lo) | (y > hi)).astype(float), np.nan)
        alpha_trace[t] = a_t.mean(axis=1)
    aci.cap_fraction = (capped.sum(axis=1) / (N * S)).tolist()      # per horizon
    return lo_out, hi_out, alpha_trace


# ----------------------------------------------------------------------- driver
def evaluate_model(name, data, alpha, gamma):
    vp, vy = data["val"]
    tp, ty = data["test"]
    # Missing readings are 0 in the data, but after the authors' scaler round
    # trip in evaluate() they come back as tiny residues, not exact zeros.
    # Treat |v| < 1e-3 mph as missing.
    for split, arr in (("val", vy), ("test", ty)):
        near = (np.abs(arr) < 1e-3) & (arr != 0)
        print(f"  [{name}/{split}] missing readings: {np.mean(np.abs(arr) < 1e-3):.2%} "
              f"(of which float residues, not exact 0: {near.mean():.2%})")
    vy = np.where(np.abs(vy) < 1e-3, np.nan, vy)
    ty = np.where(np.abs(ty) < 1e-3, np.nan, ty)
    val, test = (vp, vy), (tp, ty)
    med = tp[..., 1]

    intervals = {"raw": (tp[..., 0], tp[..., 2])}
    lo, hi, q_global = cqr(val, test, alpha, per_sensor=False)
    intervals["cqr"] = (lo, hi)
    lo, hi, q_sensor = cqr(val, test, alpha, per_sensor=True)
    intervals["cqr_sensor"] = (lo, hi)
    lo, hi, alpha_trace = aci(val, test, alpha, gamma)
    intervals["aci"] = (lo, hi)

    rows, sensor_cov = [], {}
    for method, (lo, hi) in intervals.items():
        # averaged over all 12 horizons (the authors' reporting convention)
        r = interval_metrics(lo, med, hi, ty, alpha)
        rows.append({"model": name, "method": method, "horizon": "avg_1-12", **r})
        for h in range(12):
            r = interval_metrics(lo[h], med[h], hi[h], ty[h], alpha)
            tag = f"{5 * (h + 1)}min"
            rows.append({"model": name, "method": method, "horizon": tag, **r})
        sensor_cov[method] = per_sensor_coverage(lo, hi, ty)
    extra = {
        "cqr_correction_mph_by_horizon": q_global.ravel().tolist(),
        "cqr_sensor_correction_mph_median_by_horizon":
            np.median(q_sensor[:, 0, :], axis=1).tolist(),
        "aci_final_mean_alpha_by_horizon": alpha_trace[-1].tolist(),
        "aci_fraction_of_forecasts_capped_by_horizon": aci.cap_fraction,
    }
    return rows, sensor_cov, alpha_trace, extra


COLORS = {"raw": "#2a78d6", "cqr": "#eb6834", "cqr_sensor": "#1baf7a", "aci": "#6250d6"}
LABELS = {"raw": "Raw quantiles", "cqr": "CQR (per horizon)",
          "cqr_sensor": "CQR (per horizon and sensor)", "aci": "ACI (online)"}


def plot(name, rows, sensor_cov, alpha, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#888",
                         "axes.grid": True, "grid.color": "#e4e4e4",
                         "grid.linewidth": 0.6})
    mins = np.arange(5, 65, 5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in COLORS:
        rr = [r for r in rows if r["method"] == method and r["horizon"] != "avg_1-12"]
        axes[0].plot(mins, [r["coverage"] for r in rr], marker="o", ms=5, lw=2,
                     color=COLORS[method], label=LABELS[method])
        axes[1].plot(mins, [r["width"] for r in rr], marker="o", ms=5, lw=2,
                     color=COLORS[method], label=LABELS[method])
    axes[0].axhline(1 - alpha, color="#555", lw=1, ls="--")
    axes[0].text(mins[-1], 1 - alpha, f" target {1 - alpha:.0%}", va="bottom",
                 ha="right", color="#555", fontsize=9)
    axes[0].set(xlabel="Forecast horizon (min)", ylabel="Coverage of 95% interval",
                title="Coverage on test split")
    axes[1].set(xlabel="Forecast horizon (min)", ylabel="Mean interval width (mph)",
                title="Interval width on test split")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(mins)
    fig.suptitle(name, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    f1 = os.path.join(out, f"{name}_coverage_width.png")
    fig.savefig(f1, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in COLORS:
        c = np.sort(sensor_cov[method])
        ax.plot(np.linspace(0, 1, len(c)), c, lw=2, color=COLORS[method],
                label=LABELS[method])
    ax.axhline(1 - alpha, color="#555", lw=1, ls="--")
    ax.set(xlabel="Sensors, sorted by coverage (fraction)",
           ylabel="Coverage (all horizons)", title=f"{name}: per-sensor coverage")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    f2 = os.path.join(out, f"{name}_per_sensor_coverage.png")
    fig.savefig(f2, dpi=160)
    plt.close(fig)
    return [f1, f2]


def write_csv(rows, path):
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def print_summary(rows):
    print(f"\n{'model':<22}{'method':<12}{'horizon':<10}{'MAE':>7}{'cov':>8}"
          f"{'width':>8}{'MIS':>8}{'cov<40':>8}{'sens p5':>9}")
    for r in rows:
        if r["horizon"] in ("avg_1-12", "15min", "30min", "60min"):
            print(f"{r['model']:<22}{r['method']:<12}{r['horizon']:<10}"
                  f"{r['MAE']:>7.3f}{r['coverage']:>8.3f}{r['width']:>8.2f}"
                  f"{r['MIS']:>8.2f}{r['cov[congested <40 mph]']:>8.3f}"
                  f"{r['sensor_cov_p05']:>9.3f}")


def run(models, out, alpha, gamma, make_plots=True):
    os.makedirs(out, exist_ok=True)
    all_rows, extras = [], {}
    for name, path in models:
        print(f"\n== {name}: {path}")
        data = load_npz(path, name) if isinstance(path, str) else path
        rows, sensor_cov, alpha_trace, extra = evaluate_model(name, data, alpha, gamma)
        all_rows += rows
        extras[name] = extra
        if make_plots:
            for f in plot(name, rows, sensor_cov, alpha, out):
                print("  wrote", f)
    csv_path = os.path.join(out, "coverage_conformal_metrics.csv")
    write_csv(all_rows, csv_path)
    with open(os.path.join(out, "coverage_conformal_extra.json"), "w") as f:
        json.dump({"alpha": alpha, "aci_gamma": gamma, "models": extras}, f, indent=1)
    print_summary(all_rows)
    print("\nwrote", csv_path)
    return all_rows


# --------------------------------------------------------------------- selftest
def synthetic(seed=0, n_val=1676, n_test=3376, S=40, shrink=0.6, shift_test=1.0):
    """Heteroscedastic Gaussian truth around a 'median' forecast. The model's
    interval is too narrow by `shrink`; the test period has `shift_test` times
    the noise of the validation period (distribution shift)."""
    rng = np.random.default_rng(seed)
    H = 12
    offset = rng.normal(0, 3, (1, 1, S))
    sensor_noise = 0.5 + 1.5 * rng.random((1, 1, S))   # fixed per sensor
    def split(N, scale):
        base = 55 + 10 * np.sin(np.linspace(0, 40, N))[None, :, None] + offset
        h_noise = (1 + 0.25 * np.arange(1, H + 1))[:, None, None]
        sigma = h_noise * sensor_noise * scale
        med = np.broadcast_to(base, (H, N, S)).copy()
        y = med + rng.normal(0, 1, (H, N, S)) * sigma
        # the model knows the horizon effect but not which sensors are noisy
        z = 1.959964 * h_noise * sensor_noise.mean() * shrink * np.ones((1, 1, S))
        pred = np.stack([med - z, med, med + z], -1)
        y[rng.random(y.shape) < 0.01] = 0          # missing readings
        return pred, y
    return {"val": split(n_val, 1.0), "test": split(n_test, shift_test)}


def selftest(out):
    print("Self-test 1: exchangeable data, raw intervals 40% too narrow")
    rows = run([("synthetic_iid", synthetic(shift_test=1.0))], out, 0.05, 0.005,
               make_plots=True)
    get = lambda m, h="avg_1-12": next(r for r in rows if r["method"] == m and r["horizon"] == h)
    assert get("raw")["coverage"] < 0.85, "raw should under-cover"
    for m in ("cqr", "cqr_sensor", "aci"):
        c = get(m)["coverage"]
        assert 0.94 <= c <= 0.96, f"{m} coverage {c:.3f} not ~0.95"
    assert get("cqr_sensor")["sensor_cov_p05"] > get("cqr")["sensor_cov_p05"]
    print("\nSelf-test 2: test period 30% noisier than validation (shift)")
    rows = run([("synthetic_shift", synthetic(shift_test=1.3))], out, 0.05, 0.005,
               make_plots=False)
    assert get("cqr")["coverage"] < 0.93, "static CQR should lose coverage under shift"
    assert get("aci")["coverage"] > get("cqr")["coverage"] + 0.02, "ACI should adapt"
    print("\nself-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", action="append", default=[],
                    help="name=path to a prediction .npz (repeat for each model)")
    ap.add_argument("--out", default="coverage_out")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--aci-gamma", type=float, default=0.005)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.out)
        sys.exit(0)
    if not a.npz:
        ap.error("give at least one --npz name=path, or --selftest")
    models = [tuple(s.split("=", 1)) if "=" in s
              else (os.path.splitext(os.path.basename(s))[0], s) for s in a.npz]
    run(models, a.out, a.alpha, a.aci_gamma)
