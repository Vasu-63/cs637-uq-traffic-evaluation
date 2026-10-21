"""
Load one of the authors' trained model variants outside their supervisor,
so we can run it at any batch size, on CPU or GPU, on inputs we modify.

Used by latency.py and sensor_failure.py (CS637 project).

Quirks of the authors' code handled here (read from uq4phy_traffic_network):
- Model files read relative paths such as data/sensor_graph/PEMSD4_matrix.npy,
  so we chdir into the variant folder.
- `device` is fixed when their modules are imported, so CPU-only runs must set
  CUDA_VISIBLE_DEVICES="" before this module imports torch (latency.py does).
- DCGRU cells create their weights lazily on the first forward pass (in every
  variant), and GPDE layers do the same with `h_input`. GPDE encoder and
  decoder also hard-code `self.batch_size = 256`. We run one dummy forward at
  batch 256 before loading weights, then set batch_size per call.
- PyTorch >= 2.6 needs weights_only=False for these checkpoints.
"""
import glob
import logging
import os
import sys

import numpy as np
import torch
import yaml


def find_config(variant_dir):
    pre = sorted(glob.glob(os.path.join(variant_dir, "data/model/pretrained/*/config.yaml")))
    if pre:
        return pre[0]
    own = sorted(glob.glob(os.path.join(variant_dir, "data/model/*.yaml")))
    if not own:
        raise FileNotFoundError(f"no config yaml under {variant_dir}/data/model")
    return own[0]


def latest_checkpoint_epoch(variant_dir):
    eps = [int(os.path.basename(p)[3:-4]) for p in glob.glob(os.path.join(variant_dir, "models/epo*.tar"))]
    if not eps:
        raise FileNotFoundError(f"no models/epo*.tar in {variant_dir}")
    return max(eps)


def set_batch_size(model, b):
    for part in (model.encoder_model, model.decoder_model):
        if hasattr(part, "batch_size"):
            part.batch_size = b


def is_gpde(model):
    return hasattr(model.encoder_model, "h_input")


def load_variant(variant_dir, epoch=None, config=None):
    """Returns (model, cfg, info). The model is in eval mode on the device the
    authors' modules picked (cuda if visible, else cpu)."""
    variant_dir = os.path.abspath(variant_dir)
    cfg_path = config or find_config(variant_dir)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if epoch is None:
        epoch = int(cfg.get("train", {}).get("epoch", 0)) or latest_checkpoint_epoch(variant_dir)
    ckpt_path = os.path.join(variant_dir, "models", f"epo{epoch}.tar")

    # import the authors' modules from this variant (each variant has its own copy)
    for mod in [m for m in list(sys.modules) if m == "lib" or m.startswith(("lib.", "model"))]:
        del sys.modules[mod]
    sys.path.insert(0, variant_dir)
    os.chdir(variant_dir)
    from lib import utils                       # noqa: E402
    from model.pytorch.dcrnn_model import DCRNNModel, device   # noqa: E402

    _, _, adj_mx = utils.load_graph_data(cfg["data"]["graph_pkl_filename"])
    mk = cfg["model"]
    logger = logging.getLogger("p3")
    model = DCRNNModel(adj_mx, logger, **mk).to(device)
    model.eval()

    n, din = int(mk["num_nodes"]), int(mk["input_dim"])
    # One dummy pass creates every lazily-built parameter (the DCGRU weights in
    # all variants, plus h_input in GPDE ones), as the authors' _setup_graph does.
    with torch.no_grad():
        set_batch_size(model, 256)
        model(torch.zeros(int(mk["seq_len"]), 256, n * din, device=device))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    info = {"variant_dir": variant_dir, "config": cfg_path, "checkpoint": ckpt_path,
            "epoch": epoch, "device": str(device), "gpde": is_gpde(model),
            "num_nodes": n, "input_dim": din, "seq_len": int(mk["seq_len"]),
            "horizon": int(mk["horizon"]), "adj_mx": adj_mx}
    return model, cfg, info


@torch.no_grad()
def predict(model, x_scaled, batch_size=256):
    """x_scaled: (N, 12, S, D) numpy, feature 0 already standardised (as the
    authors' load_dataset does). Returns (12, N, S, 3) in scaled units."""
    dev = next(model.parameters()).device
    N, T, S, D = x_scaled.shape
    outs = []
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x_scaled[i:i + batch_size])).float()
        b = xb.shape[0]
        set_batch_size(model, b)
        xb = xb.permute(1, 0, 2, 3).reshape(T, b, S * D).to(dev)
        out = model(xb)                         # (12, b, S, 3) for the quantile heads
        outs.append(out.reshape(out.shape[0], b, S, -1).cpu().numpy())
    set_batch_size(model, 256)
    return np.concatenate(outs, axis=1)
