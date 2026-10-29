# Implementation: our physics loss, the METR-LA check, and two new ideas

This folder holds our own implementation of the paper's physics-informed loss and the experiments built on it.
All numbers here come from files in this folder (`results/`, `logs/`) or from the paper's Tables 1 and 2.

PEMSD8 comparisons use 3 seeds, METR-LA uses 2.

## Contents

| Path | What |
|---|---|
| `physics_losses.py` | Our physics loss: drop-in replacement for the authors' `underwood_loss`, any batch size / sensor count, Underwood or Greenshields law |
| `tests/test_physics_losses.py` | 11 pytest tests, incl. loss and gradient equality with the authors' function |
| `fit_params.py` | Per-sensor fits of both laws on the PEMSD8 training split |
| `eval_run.py` | Per-horizon evaluation of one trained run (MAE, RMSE, MIS, interval width, coverage) |
| `params/` | Fitted per-sensor parameters (170 values each) and fit quality per sensor |
| `results/` | Per-horizon metrics of every run, summaries over seeds |
| `figures/` | Figures used in the slides |
| `conformal/` | Calibration metrics (raw, CQR, per-sensor CQR, ACI) for the six models of Experiment 5 |
| `logs/` | Training logs of every run (best epoch = last `saving to models/epoN.tar` line) |
| `implementation_colab.ipynb` | Colab notebook that runs the whole pipeline: setup, tests, training runs, evaluation, summaries, calibration check |

## Setup used for all runs

- Authors' code: https://github.com/tianshubao1/uq4phy_traffic_network, variant folders copied (never trained in place).
- Data: PEMSD8 and METR-LA from figshare (links in the top-level README), into `/content/data/<name>/`.
- Hardware: Google Colab, free T4 GPU; about 1.0-1.1 min per PEMSD8 epoch when one run has the GPU to itself.
- Training command (inside a run folder):
  `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml`
- `eval_run.py` uses the per-horizon metric script `horizon_metrics.py` from the reproduction folder. By default it looks in
  `reproduction/scripts/`; pass `--metrics_scripts <folder>` if it is elsewhere.
- `tests/test_physics_losses.py` compares our loss with the authors' `underwood_loss`, so the authors' code must be
  importable when the tests run (see the top of the test file for the path it expects).
- Evaluation command:
  `python eval_run.py --run <run folder> --epoch <best epoch> --data_dir /content/data/PEMSD8 --label <name> --csv results/horizon_metrics.csv`
- Seeds: the authors hard-code `random_seed = 0` in `model/pytorch/dcrnn_supervisor.py`; for seeds 1 and 2 we change only that line. The same seed means the same initialisation, so comparisons between losses are paired.
- "Single horizon" = error at exactly 15, 30 or 60 min ahead. "Averaged" = mean over all horizons up to that time (5 min ... t), which is what the authors' logs report (see Experiment 2).

## Experiment 1: our own physics loss equals the authors'

**What it tests.** Whether the paper's Underwood loss, v = v_f · exp(−k / k_m), can be re-implemented cleanly and used as a drop-in replacement.

**Change to the authors' folder** (`gpde_quantile_phy_model_PEMSD8`, λ = 0.002): one import line in `model/pytorch/dcrnn_supervisor.py`,
`from model.pytorch.physics_losses import underwood_loss`, with `physics_losses.py` copied into `model/pytorch/`.

**Result.** All 11 tests pass: same loss value and same gradients as the authors' function on identical random inputs (rtol 1e-5).
Unlike the authors' version, ours stores one parameter per sensor instead of a (12, 128, 170) copy, loads the files once
instead of every batch, and works for any batch size or sensor count.

```
pytest tests/test_physics_losses.py
```

## Experiment 2: reproducing Table 1 (PEMSD8, GPDE+Quantile+Phy)

**Runs.** `pemsd8_underwood_ours` (seeds 0, 1, 2), λ = 0.002, 50 epochs, best epochs 30 / 48 / 46.
Table 1 used λ = 0.002 too: its PEMSD8 "60 min" row is identical to the λ = 0.002 row of Table 2.

**Result (test set, mean of 3 seeds; MAE / MIS / RMSE).**

| | Paper, Table 1 | Ours, averaged up to t | Ours, single horizon |
|---|---|---|---|
| 15 min | 1.024 / 12.06 / 2.162 | 1.060 / 12.58 / 2.211 | 1.249 / 17.53 / 2.665 |
| 30 min | 1.236 / 18.50 / 2.774 | 1.268 / 19.06 / 2.805 | 1.574 / 29.16 / 3.544 |
| 60 min | 1.521 / 28.81 / 3.539 | 1.555 / 29.41 / 3.569 | 2.017 / 46.80 / 4.576 |

- The reproduction succeeds within 1-4% on every metric **if** Table 1 reports averages over horizons up to t.
- Read as errors at 15/30/60 min, Table 1 understates the 60-min MAE by about a quarter (2.02 vs 1.52) and the 60-min MIS by about 40%.
- The 95% interval covers only 81.0% / 70.1% / 57.6% of true speeds at 15 / 30 / 60 min, while the interval gets narrower (3.50 → 3.14 → 2.78 mph).

Files: `results/horizon_metrics.csv` (rows `pemsd8_underwood_ours*`), `results/greenshields_vs_underwood_summary.csv` (Underwood rows).

## Experiment 3: on METR-LA the physics loss uses time of day as "density"

**What it tests.** METR-LA has no occupancy. Its second input feature is the time of day (0-1), and the authors' loss reads it as density.

**Checks** (`figures/metrla_feature1_checks.txt`, `figures/metrla_physics_params_checks.txt`, `figures/metrla_feature1_is_time_of_day.png`):
- feature 1 is identical across all 207 sensors and equals (minute of day) / 1440;
- the shipped METR-LA parameters are unphysical: k_m < 0 for 147 of 207 sensors, and the physics target lies below every sensor's mean observed speed (median gap 20.3 mph).

**Runs.** Authors' `gpde_quantile_model` and `gpde_quantile_phy_model` (λ = 0.0005), 25 epochs, seeds 0 and 1;
best epochs 23 / 21 (seed 0) and 21 / 24 (seed 1).

**Result (test, single horizon, mean of 2 seeds, without → with physics loss).**

| | 15 min | 30 min | 60 min |
|---|---|---|---|
| MAE | 2.924 → 2.932 | 3.479 → 3.496 | 4.376 → 4.387 |
| MIS | 33.56 → 33.13 | 52.27 → 51.51 | 84.77 → 83.41 |
| Coverage | 90.1% → 89.5% | 84.9% → 84.0% | 77.8% → 76.7% |

No benefit. MAE is equal or slightly worse and coverage slightly lower with the physics term in every seed at every horizon.
Seed 0 alone showed 1-5% lower RMSE and MIS with physics; seed 1 did not (at 60 min it was worse), so that was noise.
Files: `results/metrla_tod.csv`, `results/metrla_tod_summary.csv`, `results/metrla_phy_minus_nophy_paired.csv`.

## Experiment 4 (new idea): Greenshields instead of Underwood

**What it tests.** Whether the paper's choice of the Underwood law matters, by swapping in the simpler, linear Greenshields law,
v = v_f · (1 − k / k_jam).

**Fit on the training data** (`fit_params.py`, `params/fit_quality.csv`, `figures/pemsd8_greenshields_vs_underwood_fit.png`):
Greenshields fits better on 162 of 170 sensors (median RMSE 2.44 vs 2.64 mph).

**Runs.** `pemsd8_greenshields` (seeds 0, 1, 2), same as Experiment 2 except
`PHYS_LAW=greenshields PHYS_VF=params/greenshields_v_f.npy PHYS_K=params/greenshields_k_jam.npy`. Best epochs 45 / 46 / 39.

**Result (test, single horizon, mean ± std over 3 seeds; Underwood / Greenshields).**

| | 15 min | 30 min | 60 min |
|---|---|---|---|
| MAE | 1.249 ± 0.018 / 1.254 ± 0.036 | 1.574 ± 0.016 / 1.577 ± 0.024 | 2.017 ± 0.023 / 2.009 ± 0.022 |
| MIS | 17.53 / 17.91 | 29.16 / 29.36 | 46.80 / 46.20 |
| Coverage | 81.0% / 80.0% | 70.1% / 70.2% | 57.6% / 59.3% |

The two laws are indistinguishable: every per-seed difference changes sign across seeds.
Seed 0 alone made Greenshields look clearly better at 60 min (MIS −7%, coverage +7 points); seed 1 reversed it.
This is why we report three seeds, which the paper does not.
Files: `results/greenshields_vs_underwood.csv`, `results/greenshields_vs_underwood_summary.csv`,
`results/greenshields_minus_underwood_paired.csv`, `figures/pemsd8_greenshields_vs_underwood_3seeds.png`.

## Experiment 5 (new idea): time-aligned physics loss

**What it tests.** In the authors' code the loss pairs each predicted speed with the occupancy from the model *input*,
which is exactly 60 minutes earlier than the predicted time. The speed-density law should hold at the same time and place.
We pass the true future occupancy from the training labels instead. This is used only in the training loss, so there is
no leakage: the physics loss plays no part at test time.

**Change to the run folder** (two lines in `dcrnn_supervisor.py`, `physics_losses.py` unchanged):
- in `_prepare_data`: keep the labels before they are cut down to speed, `self._y_full = y.to(device)`;
- in `_compute_loss`: pass `self._y_full` instead of `x_true` when `PHYS_ALIGN=future`.

We checked that label step j equals the input of the sample 12 steps later (same time), on the training data.

**Runs.** `pemsd8_underwood_aligned` (seeds 0, 1, 2; best epochs 49 / 48 / 44), λ = 0.002, 50 epochs, run with `PHYS_ALIGN=future`.
Compared with `pemsd8_underwood_ours` of the same seed.

**Result (test, single horizon, mean ± std over 3 seeds; plain Underwood → time-aligned).**

| | 15 min | 30 min | 60 min |
|---|---|---|---|
| MAE | 1.249 → 1.238 | 1.574 → 1.564 | 2.017 → 2.012 |
| MIS | 17.53 ± 0.59 → 16.94 ± 0.66 | 29.16 ± 0.22 → 27.95 ± 0.73 | 46.80 ± 2.34 → 44.11 ± 1.78 |
| Coverage | 81.0% → 81.7% | 70.1% → 73.0% | 57.6% → 63.8% |
| Width (mph) | 3.50 → 3.68 | 3.14 → 3.36 | 2.78 → 3.13 |

- **Interval score is lower in every seed at every horizon** (9 of 9 paired comparisons; at 60 min by 3.8 / 1.0 / 3.3).
- **60-min coverage is higher in every seed** (+6.9 / +1.4 / +10.1 points); at 15 and 30 min seed 1 is about 1 point lower.
- **No effect on accuracy:** the MAE difference changes sign between seeds; 60-min RMSE is slightly higher in all seeds (+0.004 to +0.053).
- Part of the coverage gain comes from wider intervals, and coverage is still far below the nominal 95%.

So the fix gives a modest but consistent improvement in interval quality, unlike the Greenshields swap, whose differences
flip sign between seeds. Files: `results/time_aligned_vs_underwood.csv`, `results/time_aligned_vs_underwood_summary.csv`,
`results/time_aligned_minus_underwood_paired.csv`, `figures/pemsd8_time_aligned_vs_underwood_3seeds.png`.
(Seed 1 needed three attempts because Colab cut the session; only the complete 50-epoch run is used.)

**Does the gain survive calibration?** Wider intervals alone can raise coverage, so we applied the team's conformal
calibration (`extensions/coverage_conformal.py`: CQR, per-sensor CQR and ACI, validation split as calibration set) to all six
models and compared them at the same 95% coverage. Standard interval score, all 12 horizons, mean of 3 seeds:

| Method | Authors' loss | Time-aligned | Coverage in congestion (<40 mph): authors' → time-aligned |
|---|---|---|---|
| Raw intervals | 27.86 | 26.51 (−4.9%) | 47.9% → 54.0% |
| CQR | 20.04 | 19.82 (−1.1%) | 59.5% → 63.8% |
| Per-sensor CQR | 17.67 | 17.54 (−0.7%) | 70.8% → 73.4% |
| ACI | 17.79 | 17.69 (−0.6%) | 78.2% → 79.8% |

After calibration the overall advantage almost disappears (under 1.1%, and not in every seed), the same pattern the team
found for the authors' physics loss itself. But the time-aligned model keeps **higher coverage in congested traffic**
in 47 of 48 seed × method × horizon comparisons: the physics law helps most where speed really depends on density.
Files: `conformal/coverage_conformal_metrics.csv`, `results/time_aligned_conformal_summary.csv`,
`figures/pemsd8_time_aligned_conformal.png`. To reproduce: run `eval_run.py ... --save_npz <file>` for each checkpoint,
then `python coverage_conformal.py --npz name=<file> ... --out conformal`.

## Pitfalls we hit (useful for anyone re-running this)

1. **`evaluate()` does not return predictions** in these variants (the dictionary is commented out in its `return`).
   `eval_run.py` re-enables it at runtime without editing the authors' files.
2. **Missing readings on METR-LA.** They are 0 in the data but come back from `evaluate()` as tiny non-zero numbers,
   so a `truth != 0` mask misses them (per-horizon MAE about 10 instead of 3). `eval_run.py` sets |truth| < 0.001 to 0.
   PEMSD8 has no missing readings.
3. **The authors' evaluation pads the last test batch** with copies of the final sample, so their averaged numbers differ
   slightly (0.3-1.4%) from ours, which use the real test samples only.

## Not done / future work

- METR-LA trained for the full 50 epochs (we used 25 for time).
- More seeds and other λ values for the time-aligned loss; MAEMIS and EDL variants with it.
- A density input for METR-LA (it has none), which would be needed for the physics loss to mean anything there.
