# Reproduction

Can the paper's Table 1 be reproduced from the authors' own code, and does the physics loss's gain hold up across random seeds?

**Short answer.** The numbers reproduce; the claimed benefit of the physics loss does not.

- **PEMSD4.** Evaluating the authors' six shipped checkpoints gives their logged numbers to 3–4 decimals. Once the authors' batch padding is added back, we match Table 1's 30- and 60-minute rows within 0.03–0.2%.
- **PEMSD8.** Retrained from scratch, GPDE+Quantile matches Table 1 within 0.1–0.3% on MAE and RMSE (3 seeds). GPDE+Quantile+Phy is within 2–4% and plain Quantile within 3–6%.
- **Seed check.** The paper's physics gain (MAE 1.041 → 1.024) is smaller than our seed-to-seed standard deviation (about 0.02–0.03). Over 5 paired seeds, two of them run with the authors' own physics code, the physics loss is better in only 1 of 5 seeds at each horizon. It makes 60-minute intervals slightly worse in all 5.
- **How Table 1 reports horizons.** Table 1's "15 / 30 / 60 min" numbers are averages over all steps up to that time, not the error at that horizon. Read as single-horizon errors, Table 1 understates the 60-minute error by about a quarter.

## Results

All metrics are on the test set. MIS is the authors' version, which adds |y − median| to the standard interval score (see "Issues found"). "Averaged" means the mean over forecast steps 1–3, 1–6 and 1–12 for 15, 30 and 60 minutes.

### PEMSD4: the authors' shipped checkpoints

We loaded each model at the epoch its config names and evaluated it with the authors' `evaluate()`.

| Model | Epoch | Table 1 "60 min" MAE | Authors' training log | Our evaluation |
|---|---|---|---|---|
| Quantile | 18 | 2.670 | 2.6702 | 2.6707 |
| GPDE+Quantile | 18 | 2.133 | 2.1338 | 2.1362 |
| GPDE+Quantile+Phy (λ 0.005) | 19 | 2.018 | 2.0189 | 2.0194 |
| MAEMIS | 18 | **3.167** | 2.6805 | 2.6831 |
| GPDE+MAEMIS | 16 | 2.098 | 2.0986 | 2.1010 |
| GPDE+MAEMIS+Phy (λ 0.005) | 19 | 2.044 | 2.0448 | 2.0461 |

The log and our numbers are averages over all 12 horizons, which is what Table 1 prints as "60 min". MIS and MSE match equally closely; see `results/pemsd4_table1_vs_authors_logs.csv`. The MAEMIS entry in Table 1 is a copying error: its MIS and MSE match the log exactly, but its MAE does not.

**Which reading Table 1 uses.** The figures below are the mean |% difference| from Table 1 over MAE, MIS and MSE.

| Model | 15 min (closest reading) | 30 min, padded average | 60 min, padded average |
|---|---|---|---|
| Quantile | 3.43 (single step) | 0.04 | 0.03 |
| GPDE+Quantile | 1.45 (single step) | 0.11 | 0.17 |
| GPDE+Quantile+Phy | 1.55 (single step) | 0.08 | 0.04 |
| MAEMIS | 3.60 (single step) | 0.05 | 5.13 |
| GPDE+MAEMIS | 1.07 (single step) | 0.20 | 0.15 |
| GPDE+MAEMIS+Phy | 0.25 (average) | 0.05 | 0.09 |

Without the padding, the 30- and 60-minute gaps are 2–7%. The 15-minute rows match no single reading consistently. Source: `results/pemsd4_padded_vs_table1.csv`.

**Coverage of the nominal 95% interval** (single horizon, `results/pemsd4_horizons_vs_table1.csv`):

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Quantile | 69.6% | 64.6% | 57.2% |
| GPDE+Quantile | 78.4% | 64.6% | 51.8% |
| GPDE+Quantile+Phy | 83.7% | 74.3% | 65.3% |
| MAEMIS | 59.0% | 51.6% | 45.0% |
| GPDE+MAEMIS | 75.7% | 63.3% | 50.1% |
| GPDE+MAEMIS+Phy | 84.0% | 73.7% | 63.9% |

The extensions workstream gets the same values with independent code (`extensions/`).

### PEMSD8: retrained from scratch

Each run used the authors' folder and config: 50 epochs, batch 128, λ = 0.002 for Phy. The only change was the random seed. Values are means over seeds, in the averaged reading (`results/pemsd8_trio_vs_table1.csv`).

| Model | Seeds | MAE 15 / 30 / 60 min (ours) | Table 1 | Diff |
|---|---|---|---|---|
| Quantile | 1 | 1.276 / 1.448 / 1.717 | 1.202 / 1.386 / 1.666 | +3.1 to +6.1% |
| GPDE+Quantile | 3 | 1.044 / 1.255 / 1.542 | 1.041 / 1.251 / 1.540 | +0.1 to +0.3% |
| GPDE+Quantile+Phy | 3 | 1.060 / 1.268 / 1.555 | 1.024 / 1.236 / 1.521 | +2.2 to +3.5% |

The GPDE+Quantile+Phy runs come from the implementation workstream (`implementation/`): the same seeds 0, 1 and 2, and a physics loss tested equal to the authors' `underwood_loss`. RMSE for GPDE+Quantile is 2.172 / 2.778 / 3.566 against the paper's 2.175 / 2.780 / 3.565. In the single-horizon reading, every model is 18–33% above Table 1 in MAE.

### Seed check: is the physics gain bigger than noise?

The comparison is GPDE+Quantile against GPDE+Quantile+Phy, paired by seed. The same seed gives the same initial weights, so only the loss differs. Differences are Phy minus GPDE, single horizon (`results/pemsd8_seed_check.csv`).

| Metric | 15 min | 30 min | 60 min |
|---|---|---|---|
| MAE (lower is better) | +0.026 / +0.031 / −0.012 | +0.019 / +0.033 / −0.021 | +0.024 / +0.058 / −0.033 |
| MIS (lower is better) | +0.148 / +1.234 / −0.283 | +0.380 / +1.188 / +0.068 | +1.545 / +2.108 / +1.051 |
| Coverage (higher is better) | +2.0 / −2.2 / +1.4 pts | −0.4 / −1.5 / −0.2 pts | −3.7 / −2.6 / −2.9 pts |

Seed-to-seed standard deviations of MAE are 0.016–0.025. Every mean paired difference in MAE, RMSE, MIS and coverage is below one seed standard deviation, between 0.2 and 0.9 of it; the one exception is 15-minute RMSE at 1.1, where physics is worse. So the physics loss gives no measurable gain.

At 30 and 60 minutes, the intervals are slightly worse with physics in all three seeds: MIS is higher and coverage lower. This is consistent in direction but smaller than the seed spread.

**Where the headline gain comes from** (MAE, averaged reading, 15 min):

| | Quantile → GPDE | GPDE → GPDE+Phy |
|---|---|---|
| Paper | −13.4 points (1.202 → 1.041) | −1.4 points (1.041 → 1.024) |
| Ours | −18.1 points (1.276 → 1.044) | **+1.2 points** (1.044 → 1.060) |

All of the improvement comes from the GPDE layer, which is the authors' earlier work. Our plain Quantile baseline is a single seed, which inflates the GPDE share; that does not affect the physics result.

All of these tables are also in `results/reproduction_results.xlsx`.

### Rerun: the authors' own physics code, new seeds

The seed check above pairs our GPDE+Quantile runs with physics runs that used the implementation workstream's re-implemented loss. To rule out any difference between the two implementations, we trained both models again on two new seeds, 3 and 4. This time the physics runs used **the authors' own folder** `gpde_quantile_phy_model_PEMSD8` (λ = 0.002), unchanged except for the seed (`notebooks/pemsd8_physics_rerun.ipynb`).

**Rerun against Table 1** (averaged reading, MAE 15 / 30 / 60 min, 2 seeds):

| Model | Ours | Table 1 | Diff |
|---|---|---|---|
| GPDE+Quantile | 1.050 / 1.260 / 1.553 | 1.041 / 1.251 / 1.540 | +0.7 to +0.8% |
| GPDE+Quantile+Phy (authors' code) | 1.076 / 1.286 / 1.576 | 1.024 / 1.236 / 1.521 | +3.6 to +5.0% |

**All five pairs pooled** (seeds 0–4; Phy minus GPDE, single horizon; `results/rerun/pemsd8_rerun_seed_check_all_pairs.csv`):

| Metric | 15 min | 30 min | 60 min |
|---|---|---|---|
| MAE | +1.6%, physics better in 1 of 5 seeds | +1.0%, 1 of 5 | +0.9%, 1 of 5 |
| MIS (authors') | +2.3%, 2 of 5 | +1.5%, 1 of 5 | +2.5%, 0 of 5 (paired t-test p = 0.03) |
| Coverage | −0.3 pts, 3 of 5 | −1.1 pts, 1 of 5 | −3.1 pts, 0 of 5 (p = 0.009) |

The physics loss is worse on average for every metric at every horizon. The one exception is 60-minute RMSE (−0.6%, better in 4 of 5 seeds, p = 0.26). The only effect consistent across seeds is slightly worse 60-minute intervals. Fifteen comparisons were tested, so a p-value near 0.03 on its own is weak evidence; the 5-of-5 direction is the stronger point.

The two physics implementations behave the same: their single-horizon MAE ranges overlap at every horizon (15 min: 1.235–1.290 with the authors' code, 1.234–1.268 with the re-implementation; `results/rerun/pemsd8_rerun_phy_code_check.csv`).

Rerun runs (best epoch = lowest validation MAE): GPDE+Quantile seed 4 at epoch 47, GPDE+Quantile+Phy seed 4 at epoch 48; seed 3 epochs are in `results/rerun/pemsd8_runs/`.

## How to rerun

Everything runs in Google Colab on a free T4 GPU, with a shared Google Drive folder `MyDrive/CS637/`. Datasets, checkpoints and prediction files stay on Drive and are not committed. The notebooks download or regenerate them.

### 1. PEMSD4 checkpoint evaluation (`notebooks/pemsd4_checkpoint_evaluation.ipynb`)

About 25 minutes on the first run. Choose Runtime → T4 GPU, then Run all.

| Section | What it does |
|---|---|
| 1 | Sparse-clones [the authors' repo](https://github.com/tianshubao1/uq4phy_traffic_network): only the six PEMSD4 folders and the PEMSD8 trio. |
| 2 | Downloads `pems04.npz` from [Davidham3/ASTGCN](https://github.com/Davidham3/ASTGCN) (`data/PEMS04/`), reorders its channels, and builds train/val/test windows with `scripts/process_pems.py`. |
| 3 | **Smoke test:** evaluates GPDE+Quantile+Phy and checks it against the authors' log (MAE 2.0189). |
| 4 | Evaluates all six checkpoints (about 30 s each) and saves validation and test predictions. |
| 5–6 | Compares with the authors' logs and with Table 1, per horizon and in both readings. |
| 8 | Checks the sparse graph matrix (finding 14 below). |

Expected sample counts: 11,871 train, 1,676 val and 3,376 test windows. That is 47 batches per epoch, the same as the authors' logs.

### 2. PEMSD8 retraining and seed check (`notebooks/pemsd8_retraining_and_seed_check.ipynb`)

There are four runs of about 50 minutes each, split over two Colab accounts by setting `MY_RUNS` in section 0:

| Account | `MY_RUNS` |
|---|---|
| A | `['pemsd8_gpde_quantile_seed0', 'pemsd8_quantile_seed0']` |
| B | `['pemsd8_gpde_quantile_seed1', 'pemsd8_gpde_quantile_seed2']` |

Run all on both accounts.

| Section | What it does |
|---|---|
| 2 | Downloads the authors' processed PEMSD8 files (figshare article [28347662](https://figshare.com/articles/dataset/PEMSD8/28347662)) to Colab's local disk. |
| 4 | For each run: copies the authors' folder to local disk and sets `random_seed` in our copy of `dcrnn_supervisor.py`. Trains 50 epochs, copies checkpoints and logs to Drive, and evaluates the best checkpoint. |
| 5 | Prints tables A–D, once all four runs are done. Needs no GPU. In a fresh session, run sections 0 and 3 first. |

A run marked `DONE` on Drive is never retrained, so re-running is safe.

| Run | Model | Seed | Best epoch | Minutes | Val MAE |
|---|---|---|---|---|---|
| `pemsd8_quantile_seed0` | Quantile | 0 | 49 | 48 | 1.7110 |
| `pemsd8_gpde_quantile_seed0` | GPDE+Quantile | 0 | 43 | 51 | 1.5233 |
| `pemsd8_gpde_quantile_seed1` | GPDE+Quantile | 1 | 35 | 57 | 1.5376 |
| `pemsd8_gpde_quantile_seed2` | GPDE+Quantile | 2 | 44 | 51 | 1.5672 |

The best epoch is the one with the lowest validation MAE, which is also the last checkpoint the authors' code saves. Library versions and the authors' repo commit are printed in section 0 and section 1 of the notebooks.

## What we changed, and why

None of the authors' files are edited in their repo. Every change is made in a copy of the folder, or at runtime.

| Change | Why |
|---|---|
| PEMS04 channels reordered from [flow, occupancy, speed] to [speed, occupancy, flow] | The authors' code takes speed as feature 0 (target, normalised) and occupancy as feature 1 (physics loss). |
| Our own `process_pems.py` replaces the shipped PEMSD4 processing script | The script in the PEMSD4 folders is the PEMSD8 one (`pems8ndy.npy`, 170 sensors hard-coded). Its loop also writes `x[i:]` instead of `x[i]`: the result is the same, but it runs very slowly. Ours uses the same 70/10/20 split and 12-in/12-out windows. |
| `yaml.safe_load` in our evaluation script | `run_demo_pytorch.py` calls `yaml.load(f)`, which fails on PyYAML 6. |
| `torch.load(..., weights_only=False)` | PyTorch 2.6+ refuses the authors' checkpoint dicts otherwise. |
| `random_seed = 0` set to the run's seed, in our copy | The seed is hard-coded, so the paper's "five runs with random initialisation" cannot come from this code unchanged. |
| PEMSD8: `evaluate()`'s prediction output re-enabled at runtime | The return line ends in `#,{'prediction': ...}`, commented out. |
| PEMSD8: checkpoint placed in both `models/` and `<folder>/models/` | `save_model` writes `models/epoN.tar`, but `load_model` reads `<folder>/models/epoN.tar`, so their own checkpoints cannot be reloaded from inside the folder. |
| Our metrics drop the padded copies of the last test sample | The authors' `DataLoader` pads the last batch, and those copies are counted in their metrics. `horizon_metrics.py` reports both versions (`pad_to_batch`) so their numbers can be matched exactly. |
| Readings with \|truth\| < 1e-3 are treated as missing | Missing readings come back from `evaluate()` as float residues, not exact zeros. PEMSD4 and PEMSD8 have none, so no number here changes; it matters on METR-LA. |

## Issues found in the paper and code

The numbers refer to the team's findings list.

- **Table 1 horizons are cumulative averages (15, 35, 39, 40).** Both datasets report "15 / 30 / 60 min" as averages up to that time. On PEMSD4, the reported numbers also include duplicated padding samples.
- **Single runs, not means of five (16, 41).** The PEMSD4 rows equal one logged run to 3–4 decimals, although Section 6.1.2 says five runs were averaged. The seed is fixed in the code, and no standard deviations are reported. Our 3-seed spread is larger than the claimed physics gain.
- **The physics loss adds nothing on PEMSD8 (41, 42).** The whole headline gain comes from GPDE, which is prior work.
- **MAEMIS PEMSD4 60-min MAE typo (17):** 3.167 printed; the log gives 2.6805.
- **The reported "MIS" is not Eq. 4 (13).** `mis_loss` adds \|y − median\| to the standard interval score. We report both versions (`mis_authors`, `mis_std`).
- **Scrambled diffusion matrices (14).** `DCGRUCell._build_sparse_matrix`, inherited from DCRNN-PyTorch, re-sorts the sparse indices but not the values. On the PEMSD4 graph, the stored matrix differs from the intended one by up to 0.704, where the largest entry is 1.0, with an identical sparsity pattern. Every reported and reproduced result was trained with this bug; its effect on accuracy is untested.
- **Intervals are badly under-calibrated.** The nominal 95% intervals cover only 45–65% at 60 minutes on PEMSD4 and 50–61% on PEMSD8. The paper never measures coverage.

## Files

```
reproduction/
├── README.md
├── notebooks/
│   ├── pemsd4_checkpoint_evaluation.ipynb
│   ├── pemsd8_retraining_and_seed_check.ipynb
│   └── pemsd8_physics_rerun.ipynb       authors' own physics code, seeds 3-4
├── scripts/
│   ├── process_pems.py          raw PEMS array -> train/val/test windows
│   ├── eval_checkpoint.py       evaluate one checkpoint, save val + test predictions
│   ├── horizon_metrics.py       per-horizon metrics, both readings, optional padding
│   └── check_sparse.py          sparse graph matrix check
├── results/
│   ├── pemsd4_table1_vs_authors_logs.csv
│   ├── pemsd4_horizons_vs_table1.csv
│   ├── pemsd4_padded_vs_table1.csv
│   ├── pemsd8_runs/<run>.csv    per-run metrics, val and test, all readings
│   ├── pemsd8_all_runs_test.csv
│   ├── pemsd8_trio_vs_table1.csv
│   ├── pemsd8_seed_check.csv
│   ├── reproduction_results.xlsx
│   └── rerun/                   pemsd8_runs/<run>.csv, pemsd8_rerun_*.csv
└── logs/
    ├── <run>/                   info.log and stdout.log of each PEMSD8 run
    └── rerun/<run>/             the same for the 4 rerun runs
```
