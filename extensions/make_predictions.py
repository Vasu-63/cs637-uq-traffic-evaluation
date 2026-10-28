"""
Save validation and test predictions of the authors' PEMSD4 checkpoints in
the format coverage_conformal.py reads, straight from the checkpoints.

CS637 project. Use it when the shared prediction files in outputs/pemsd4/ are
not available yet, or as a cross-check of them: the coverage numbers from both
should agree.

Output: one <variant>.npz per model with
    val_pred / test_pred    (12, N, S, 3)  lower / median / upper, mph
    val_truth / test_truth  (12, N, S)     mph, missing readings exactly 0
No padded samples. Truth is taken from the raw data, not from the scaler round
trip, so missing readings stay exact zeros.

Usage (Colab, GPU, about a minute per model; about 200 MB per model, so write to
local disk, not the shared Drive):
    python make_predictions.py --dataset_dir /content/data/PEMSD4 \
        --variant /content/authors/quantile_model_PEMSD4 --variant ... \
        --out /content/predictions
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_utils import load_variant, predict          # noqa: E402

HERE = os.getcwd()


def main(a):
    os.makedirs(a.out, exist_ok=True)
    d = os.path.abspath(os.path.join(HERE, a.dataset_dir))
    tr = np.load(os.path.join(d, "train.npz"))["x"]
    mean, std = float(tr[..., 0].mean()), float(tr[..., 0].std())   # the authors' scaler
    del tr
    splits = {}
    for s in ("val", "test"):
        z = np.load(os.path.join(d, f"{s}.npz"))
        splits[s] = (z["x"].astype(np.float32), z["y"][..., 0].transpose(1, 0, 2).astype(np.float32))
        print(f"{s}: x {z['x'].shape}")
    for vdir in a.variant:
        vdir = os.path.abspath(os.path.join(HERE, vdir))
        name = os.path.basename(vdir.rstrip("/"))
        model, cfg, info = load_variant(vdir)
        os.chdir(HERE)
        out = {}
        for s, (x, y) in splits.items():
            xs = x.copy()
            xs[..., 0] = (xs[..., 0] - mean) / std
            out[f"{s}_pred"] = (predict(model, xs, a.batch_size) * std + mean).astype(np.float32)
            out[f"{s}_truth"] = y
        path = os.path.join(a.out, f"{name}.npz")
        np.savez_compressed(path, **out)
        with open(os.path.join(a.out, f"{name}_meta.json"), "w") as f:
            json.dump({"variant": vdir, "checkpoint": info["checkpoint"], "epoch": info["epoch"],
                       "dataset_dir": d, "scaler_mean": mean, "scaler_std": std,
                       "shapes": {k: list(v.shape) for k, v in out.items()}}, f, indent=1)
        print(f"{name} (epoch {info['epoch']}): wrote {path}")
        del model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", default="predictions_out")
    ap.add_argument("--batch_size", type=int, default=256)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    main(a)
