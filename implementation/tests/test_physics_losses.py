"""
Tests for implementation/physics_losses.py.

Run in Colab from the folder that contains `implementation/`:
    !pip -q install pytest
    !AUTHORS_DIR=/content/drive/MyDrive/CS637/uq4phy_traffic_network/gpde_quantile_phy_model_PEMSD8 \
        python -m pytest -q implementation/tests

AUTHORS_DIR must be a PEMSD8 GPDE+Quantile+Phy folder (it holds the authors'
model/pytorch/loss.py and data/sensor_graph/PEMSD8_*.npy). Without it the
equivalence tests are skipped and only the unit tests run.
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import physics_losses as pl  # noqa: E402

N = 170
torch.manual_seed(0)

# --------------------------------------------------------------------------
# Unit tests of the laws
# --------------------------------------------------------------------------

def test_underwood_at_zero_density_is_free_flow():
    v_f, k_m = torch.full((N,), 70.0), torch.full((N,), 0.5)
    assert torch.allclose(pl.underwood_speed(torch.zeros(N), v_f, k_m), v_f)


def test_underwood_at_k_m_is_v_f_over_e():
    v_f, k_m = torch.full((N,), 70.0), torch.full((N,), 0.5)
    assert torch.allclose(pl.underwood_speed(k_m.clone(), v_f, k_m), v_f / math.e)


def test_greenshields_end_points_and_clamp():
    v_f, k_jam = torch.full((N,), 70.0), torch.full((N,), 0.6)
    assert torch.allclose(pl.greenshields_speed(torch.zeros(N), v_f, k_jam), v_f)
    assert torch.allclose(pl.greenshields_speed(k_jam.clone(), v_f, k_jam), torch.zeros(N), atol=1e-6)
    assert (pl.greenshields_speed(2 * k_jam, v_f, k_jam) >= 0).all()


@pytest.mark.parametrize('law', ['underwood', 'greenshields'])
def test_perfect_prediction_gives_zero_loss(law):
    T, B = 12, 7
    v_f = 60 + 20 * torch.rand(N)
    k = 0.3 + 0.5 * torch.rand(N)
    dens = 0.4 * torch.rand(T, B, N)
    speed = pl.LAWS[law](dens, v_f, k)
    y_true = 50 + torch.rand(T, B, N)
    assert pl.physics_residual_loss(speed, dens, y_true, v_f, k, law=law).item() < 1e-10


def test_gradient_reaches_prediction():
    T, B = 12, 4
    pred = (60 + torch.rand(T, B, N)).requires_grad_()
    loss = pl.physics_residual_loss(pred, torch.rand(T, B, N) * 0.3, 50 + torch.rand(T, B, N),
                                    torch.full((N,), 70.0), torch.full((N,), 0.5))
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_missing_values_are_masked():
    T, B = 12, 4
    v_f, k = torch.full((N,), 70.0), torch.full((N,), 0.5)
    dens = torch.rand(T, B, N) * 0.3
    pred = pl.underwood_speed(dens, v_f, k)
    pred_bad = pred.clone()
    y_true = 50 + torch.rand(T, B, N)
    y_true[:, :, 0] = 0          # sensor 0 missing everywhere
    pred_bad[:, :, 0] += 100     # huge error only where the truth is missing
    assert pl.physics_residual_loss(pred_bad, dens, y_true, v_f, k).item() < 1e-10


def test_per_sensor_accepts_authors_layout():
    vals = np.arange(N, dtype=float)
    assert np.allclose(pl.per_sensor(np.broadcast_to(vals, (12, 128, N))), vals)
    with pytest.raises(ValueError):
        pl.per_sensor(np.random.rand(12, 128, N))


def test_any_batch_size_works(tmp_path, monkeypatch):
    np.save(tmp_path / 'vf.npy', np.full(N, 70.0))
    np.save(tmp_path / 'k.npy', np.full(N, 0.5))
    monkeypatch.setenv('PHYS_VF', str(tmp_path / 'vf.npy'))
    monkeypatch.setenv('PHYS_K', str(tmp_path / 'k.npy'))
    for B in (1, 37, 128):
        y_pred = 60 + torch.rand(12, B, N, 3)
        x_true = torch.rand(12, B, N * 3)
        y_true = 50 + torch.rand(12, B, N)
        assert torch.isfinite(pl.underwood_loss(y_pred, x_true, y_true))

# --------------------------------------------------------------------------
# Equivalence with the authors' implementation
# --------------------------------------------------------------------------

AUTHORS_DIR = os.environ.get('AUTHORS_DIR')
needs_authors = pytest.mark.skipif(not AUTHORS_DIR, reason='set AUTHORS_DIR to a PEMSD8 phy folder')


def _authors_loss_module():
    path = os.path.join(AUTHORS_DIR, 'model', 'pytorch', 'loss.py')
    spec = importlib.util.spec_from_file_location('authors_loss', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_batch(device, missing=True):
    g = torch.Generator().manual_seed(1)
    y_pred = (40 + 30 * torch.rand(12, 128, N, 3, generator=g)).to(device)
    x_true = torch.rand(12, 128, N, 3, generator=g)
    x_true[..., 1] *= 0.6                     # occupancy-like values
    x_true = x_true.reshape(12, 128, N * 3).to(device)
    y_true = (40 + 30 * torch.rand(12, 128, N, generator=g))
    if missing:
        y_true[:, :5, :10] = 0                # some missing truths
    return y_pred, x_true, y_true.to(device)


@needs_authors
def test_matches_authors_underwood_loss(monkeypatch):
    authors = _authors_loss_module()
    monkeypatch.chdir(AUTHORS_DIR)            # authors' loss reads relative paths
    monkeypatch.setenv('PHYS_LAW', 'underwood')
    monkeypatch.delenv('PHYS_VF', raising=False)
    monkeypatch.delenv('PHYS_K', raising=False)
    for missing in (False, True):
        y_pred, x_true, y_true = _random_batch(authors.device, missing)
        theirs = authors.underwood_loss(y_pred, x_true, y_true)
        ours = pl.underwood_loss(y_pred, x_true, y_true)
        assert torch.allclose(ours.double(), theirs.double(), rtol=1e-5), (ours.item(), theirs.item())


@needs_authors
def test_matches_authors_gradient(monkeypatch):
    authors = _authors_loss_module()
    monkeypatch.chdir(AUTHORS_DIR)
    monkeypatch.setenv('PHYS_LAW', 'underwood')
    monkeypatch.delenv('PHYS_VF', raising=False)
    monkeypatch.delenv('PHYS_K', raising=False)
    y_pred, x_true, y_true = _random_batch(authors.device)
    a = y_pred.clone().requires_grad_()
    b = y_pred.clone().requires_grad_()
    authors.underwood_loss(a, x_true, y_true).backward()
    pl.underwood_loss(b, x_true, y_true).backward()
    assert torch.allclose(a.grad.double(), b.grad.double(), rtol=1e-4, atol=1e-10)
