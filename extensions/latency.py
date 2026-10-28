"""
CPU latency, model size and memory of the authors' PEMSD4 models.

CS637 project. The paper reports no timing, memory or deployment
numbers (critique point 11); this script supplies them for the edge/CPS view:
can a roadside or traffic-centre CPU produce the 12-step forecast with a 95%
interval well inside the 5-minute sampling period?

For each variant folder it reports
  - trainable parameters (total, and the GPDE layer's own share)
  - checkpoint file size and float32 weight size
  - forward-pass latency for one forecast of the whole network
    (batch 1 = the real-time case) and for batches of 16 / 64,
    with 1 CPU thread (small embedded core) and with all threads
    (on Colab, PyTorch sees a single thread, so only the 1-thread rows appear)
  - peak resident memory of the process

Runs on CPU even when a GPU is present (CUDA is hidden before torch is
imported). Inputs are random but in realistic ranges; latency does not
depend on the values.

Usage (Colab or laptop):
    python latency.py \
        --variant /content/drive/MyDrive/CS637/extensions_out/models/quantile_model_PEMSD4 \
        --variant .../gpde_quantile_model_PEMSD4 \
        --variant .../gpde_quantile_phy_model_PEMSD4 \
        --out /content/drive/MyDrive/CS637/extensions_out/latency
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # before torch is imported

import argparse
import csv
import json
import platform
import resource
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_utils import load_variant, set_batch_size   # noqa: E402

HERE = os.getcwd()


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if platform.system() != "Darwin" else r / 1024 / 1024


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def make_input(info, b, rng):
    T, S, D = info["seq_len"], info["num_nodes"], info["input_dim"]
    x = np.zeros((T, b, S, D), np.float32)
    x[..., 0] = rng.normal(0, 1, (T, b, S))                 # standardised speed
    if D > 1:
        x[..., 1] = rng.uniform(0.0, 0.2, (T, b, S))       # occupancy (raw)
    if D > 2:
        x[..., 2] = rng.uniform(0, 600, (T, b, S))         # flow veh/5 min (raw)
    return torch.from_numpy(x.reshape(T, b, S * D))


@torch.no_grad()
def time_forward(model, x, warmup, reps):
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"median_ms": statistics.median(ts),
            "p95_ms": ts[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))],
            "min_ms": ts[0]}


def run(variants, out, batches, reps, warmup):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, sizes = [], []
    max_threads = torch.get_num_threads()
    for vdir in variants:
        vdir = os.path.abspath(os.path.join(HERE, vdir))
        name = os.path.basename(vdir.rstrip("/"))
        rss0 = peak_rss_mb()
        model, cfg, info = load_variant(vdir)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gpde_names = ("h_input", "lmbda", "beta", "scale1", "scale2")
        n_gpde = sum(p.numel() for k, p in model.named_parameters()
                     if k.split(".")[-1] in gpde_names)
        size = {
            "model": name, "epoch": info["epoch"], "gpde": info["gpde"],
            "params": n_params, "gpde_params": n_gpde,
            "checkpoint_MB": os.path.getsize(info["checkpoint"]) / 2**20,
            "weights_fp32_MB": n_params * 4 / 2**20,
            "graph_nnz": int((info["adj_mx"] > 0).sum()),
            "num_nodes": info["num_nodes"],
        }
        print(f"\n== {name}  epoch {info['epoch']}  params {n_params:,}  "
              f"(GPDE {n_gpde})  ckpt {size['checkpoint_MB']:.2f} MB")
        for threads in sorted({1, max_threads}):
            torch.set_num_threads(threads)
            for b in batches:
                set_batch_size(model, b)
                x = make_input(info, b, rng)
                t = time_forward(model, x, warmup, reps)
                row = {"model": name, "threads": threads, "batch": b, **t,
                       "per_forecast_ms": t["median_ms"] / b,
                       "share_of_5min_budget": t["median_ms"] / 1000 / 300}
                rows.append(row)
                print(f"  threads {threads:>2}  batch {b:>4}: median {t['median_ms']:8.1f} ms"
                      f"  p95 {t['p95_ms']:8.1f} ms  per forecast {row['per_forecast_ms']:7.2f} ms")
        set_batch_size(model, 256)
        torch.set_num_threads(max_threads)
        size["peak_rss_MB_after"] = peak_rss_mb()
        size["peak_rss_MB_before"] = rss0
        sizes.append(size)
        del model
        os.chdir(HERE)

    meta = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "max_threads": max_threads,
            "torch": torch.__version__, "python": platform.python_version(),
            "reps": reps, "warmup": warmup,
            "note": "one forecast = 12 x 5-min steps ahead for every sensor, with lower/median/upper"}
    for fname, data in (("latency.csv", rows), ("model_size.csv", sizes)):
        with open(os.path.join(out, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(out, "latency_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nCPU: {meta['cpu']} ({meta['logical_cpus']} logical), torch {meta['torch']}")
    print("wrote", os.path.join(out, "latency.csv"), "and model_size.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True,
                    help="path to an authors' variant folder with models/epoN.tar")
    ap.add_argument("--out", default="latency_out")
    # defaults match the run behind results/latency.csv and results/model_size.csv (batch 256 took ~10 min per model on Colab's CPU)
    ap.add_argument("--batches", default="1,16,64")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    run(a.variant, a.out, [int(b) for b in a.batches.split(",")], a.reps, a.warmup)
