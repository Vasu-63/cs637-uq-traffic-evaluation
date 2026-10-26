"""
Physics losses for speed forecasting (CS637, Person 2).

Our own implementation of the physics regulariser from Bao et al. (TCPS 2026):
    L_phy = mean( mask * (v_pred - V(k))^2 )
where V(k) is a speed-density law evaluated at the observed density/occupancy k.

Two laws are provided:
    Underwood    V(k) = v_f * exp(-k / k_m)              (the paper's law)
    Greenshields V(k) = max(0, v_f * (1 - k / k_jam))    (our new-idea comparison)

Differences from the authors' `model/pytorch/loss.py::underwood_loss`:
  * parameters are stored once per sensor, shape (N,), and broadcast; the
    authors' files are (12, 128, 170) copies, so their loss only works with
    batch size 128 on PEMSD8, and it re-reads both .npy files on every batch;
  * tensor shapes are taken from the prediction, so any batch size, horizon
    length, sensor count or feature count works;
  * the law (Underwood or Greenshields) and parameter files are chosen by
    environment variables, so the training code does not need editing.

Drop-in use inside the authors' code: change only the import line in
model/pytorch/dcrnn_supervisor.py so that `underwood_loss` comes from this
module instead of model.pytorch.loss. The signature
underwood_loss(y_pred, x_true, y_true) is the same.

Environment variables (all optional):
    PHYS_LAW      'underwood' (default) or 'greenshields'
    PHYS_VF       path to per-sensor v_f .npy   (default data/sensor_graph/PEMSD8_v_f.npy)
    PHYS_K        path to per-sensor k_m / k_jam .npy
                  (default data/sensor_graph/PEMSD8_rho_max.npy)
    PHYS_DENS_IDX input feature index used as density (default 1 = occupancy)
"""
import os
import numpy as np
import torch

# ----------------------------------------------------------------------------
# Speed-density laws
# ----------------------------------------------------------------------------

def underwood_speed(density, v_f, k_m):
    """Underwood law v = v_f * exp(-k / k_m). v_f, k_m broadcast over the last (sensor) axis."""
    return v_f * torch.exp(-density / k_m)


def greenshields_speed(density, v_f, k_jam):
    """Greenshields law v = v_f * (1 - k / k_jam), clamped at 0 for k > k_jam."""
    return torch.clamp(v_f * (1.0 - density / k_jam), min=0.0)


LAWS = {'underwood': underwood_speed, 'greenshields': greenshields_speed}

# ----------------------------------------------------------------------------
# Core loss (pure function, no file access)
# ----------------------------------------------------------------------------

def physics_residual_loss(speed_pred, density, y_true, v_f, k, law='underwood'):
    """
    speed_pred : (T, B, N) predicted speed (the median for quantile models)
    density    : (T, B, N) density/occupancy at the same positions
    y_true     : (T, B, N) true speed; zeros are treated as missing (masked out),
                 and the mask is rescaled to mean 1, exactly as in the authors' code
    v_f, k     : (N,) per-sensor parameters (or scalars)
    """
    v_law = LAWS[law](density, v_f, k)
    mask = (y_true != 0).float()
    mask = mask / mask.mean()
    loss = (speed_pred - v_law) ** 2 * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()

# ----------------------------------------------------------------------------
# Parameter loading
# ----------------------------------------------------------------------------

def per_sensor(arr):
    """
    Reduce a parameter array to one value per sensor (N,).
    Accepts (N,), or the authors' (12, 128, N) layout, checking that the
    leading axes really are copies.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    flat = arr.reshape(-1, arr.shape[-1])
    if not np.allclose(flat, flat[:1]):
        raise ValueError('parameter array is not constant over its leading axes')
    return flat[0]


_CACHE = {}


def _params(device, dtype):
    """Load (v_f, k) once per (device, dtype) from the paths in the environment."""
    vf_path = os.environ.get('PHYS_VF', 'data/sensor_graph/PEMSD8_v_f.npy')
    k_path = os.environ.get('PHYS_K', 'data/sensor_graph/PEMSD8_rho_max.npy')
    key = (vf_path, k_path, str(device), dtype)
    if key not in _CACHE:
        v_f = torch.as_tensor(per_sensor(np.load(vf_path)), device=device, dtype=dtype)
        k = torch.as_tensor(per_sensor(np.load(k_path)), device=device, dtype=dtype)
        _CACHE[key] = (v_f, k)
    return _CACHE[key]

# ----------------------------------------------------------------------------
# Drop-in replacement for the authors' underwood_loss(y_pred, x_true, y_true)
# ----------------------------------------------------------------------------

def underwood_loss(y_pred, x_true, y_true):
    """
    Same signature and semantics as the authors' function, generalised:
      y_pred : (T, B, N, 3) lower / median / upper; the median (index 1) is used
      x_true : anything that reshapes to (T, B, N, F); feature PHYS_DENS_IDX is density
      y_true : (T, B, N) true speed, used only for the missing-value mask
    The law is chosen by PHYS_LAW (default 'underwood'), despite the function name,
    so the same import swap also runs the Greenshields experiment.
    """
    law = os.environ.get('PHYS_LAW', 'underwood').lower()
    dens_idx = int(os.environ.get('PHYS_DENS_IDX', '1'))
    T, B, N = y_pred.shape[:3]
    speed_pred = y_pred[..., 1]
    density = x_true.reshape(T, B, N, -1)[..., dens_idx]
    v_f, k = _params(y_pred.device, y_pred.dtype)
    return physics_residual_loss(speed_pred, density, y_true, v_f, k, law=law)
