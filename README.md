# Evaluating "Uncertainty Quantification for Physics-Informed Traffic Graph Networks with Adversarial Attacks"

Course project for CS637 (Embedded and Cyber-Physical Systems), IIT Kanpur. The project evaluates:

> Bao, Nguyen, Liu, Ma, Johnson, Wei. *Uncertainty Quantification for Physics-Informed Traffic Graph Networks with Adversarial Attacks.* ACM Transactions on Cyber-Physical Systems, 2026. DOI [10.1145/3844727](https://doi.org/10.1145/3844727). Authors' code: [tianshubao1/uq4phy_traffic_network](https://github.com/tianshubao1/uq4phy_traffic_network).

The paper forecasts traffic speed on road-sensor graphs 15, 30 and 60 minutes ahead, with a 95% prediction interval. Its base model is DCRNN with a physics-based GPDE layer from the authors' earlier work. Its new contribution is an **Underwood physics loss** that pulls predictions toward a classical speed–density law.

**Verdict.** The authors' code runs, and its numbers reproduce closely. But the new physics loss gives no gain beyond run-to-run noise. The accuracy gains come from the GPDE layer, the intervals are not calibrated, and the robustness advantage under attack comes from a weaker attack on the physics models. Full argument: **[docs/critique.md](docs/critique.md)**.

## Headline findings

| Finding | Evidence |
|---|---|
| **The numbers reproduce.** The shipped PEMSD4 checkpoints match the authors' logs to 3–4 decimals. PEMSD8 GPDE+Quantile, retrained from scratch, matches Table 1 within 0.1–0.3% | [reproduction/](reproduction/README.md) |
| **The physics loss adds nothing.** Over 5 paired seeds, 2 of them with the authors' own physics code, physics is better in only 1 of 5 seeds at each horizon. The paper's gain (0.017 MAE) is smaller than the seed spread (0.016–0.028) | [reproduction/](reproduction/README.md#rerun-the-authors-own-physics-code-new-seeds) |
| **Table 1's "15 / 30 / 60 min" are cumulative averages, not per-horizon errors.** Read as per-horizon errors, Table 1 understates the 60-minute error by about a quarter. The tables also count padded duplicate test samples, and report single runs as means of five | [reproduction/](reproduction/README.md) |
| **The "95%" intervals are not calibrated.** They cover 52–65% of true speeds at 60 minutes, and 14–50% in congestion. Conformal calibration restores 95% | [extensions/](extensions/README.md) |
| **The robustness claim rests on an unequal attack.** Under the same PGD for every model, the physics models are no more robust than their baselines | [analysis/](analysis/) |
| **The physics loss is misaligned and, on METR-LA, not physical.** It compares each prediction with occupancy from 60 minutes earlier, and on METR-LA its "density" is the time of day. Fixing the alignment improves intervals by 3–6% | [implementation/](implementation/README.md) |
| **Dead sensors corrupt the plain model's neighbours; GPDE contains them.** A simple repair from neighbouring sensors fixes most of the damage | [extensions/](extensions/README.md) |
| **Deployment cost is negligible.** 49–58 ms per network-wide forecast on one CPU core, against a 5-minute sensor period | [extensions/](extensions/README.md) |

## The brief's four activities

| Activity | Where |
|---|---|
| Implement the technical idea | [implementation/](implementation/README.md): our own physics-loss module with 11 tests, shown equal to the authors' loss and swapped into their model |
| Reproduce the results | [reproduction/](reproduction/README.md): PEMSD4 checkpoints, PEMSD8 retraining, and the 5-seed check of the physics loss. Also [analysis/](analysis/): Table 3 traced to the authors' logs |
| Introduce new ideas | [implementation/](implementation/README.md): Greenshields instead of Underwood; a time-aligned physics loss. [extensions/](extensions/README.md): conformal calibration of the intervals |
| Introduce new use cases | [extensions/](extensions/README.md): dead and stuck sensors, CPU latency and model size. [analysis/](analysis/): Table 3 re-attacked with the same PGD for every model. [implementation/](implementation/README.md): the METR-LA time-of-day test |

## Repository map

| Folder | Contents |
|---|---|
| [`reproduction/`](reproduction/README.md) | Notebooks, scripts, per-run results and training logs for the PEMSD4 evaluation and PEMSD8 retraining; `scripts/horizon_metrics.py` and `scripts/process_pems.py` are shared with the other folders |
| [`implementation/`](implementation/README.md) | Our physics loss (`physics_losses.py`) and its tests, parameter fits, the METR-LA experiment, Greenshields and time-aligned experiments |
| [`extensions/`](extensions/README.md) | Coverage and conformal calibration, sensor-failure experiment, latency and model size |
| [`analysis/`](analysis/) | Table 3 checks: `check_table3_logs.py` traces each row to a log line (`table3_vs_logs.csv`); `table3_reattack.py` re-attacks the shipped checkpoints with the same PGD (`table3_reattack_summary.csv`) |
| [`docs/`](docs/critique.md) | The critique |

## How to rerun

Everything runs on Google Colab with a free T4 GPU. Each folder's README gives the exact steps, run times and expected outputs.

- **Authors' code:** each notebook sparse-clones only the folders it needs from [tianshubao1/uq4phy_traffic_network](https://github.com/tianshubao1/uq4phy_traffic_network). None of the authors' files are edited; changes happen in copies or at runtime, and each README lists them.
- **Data:**
  - PEMSD8: [figshare 28347662](https://figshare.com/articles/dataset/PEMSD8/28347662)
  - METR-LA: [figshare 28347830](https://figshare.com/articles/dataset/METR-LA/28347830)
  - PEMSD4: `data/PEMS04/pems04.npz` from [Davidham3/ASTGCN](https://github.com/Davidham3/ASTGCN), processed with `reproduction/scripts/process_pems.py`
- **Suggested order:**
  1. `reproduction/notebooks/pemsd4_checkpoint_evaluation.ipynb` (about 25 min, no training)
  2. `implementation/implementation_colab.ipynb`
  3. `extensions/calibration_failure_latency.ipynb`
  4. the PEMSD8 notebooks in `reproduction/notebooks/` (about 50 min per training run)

Not in the repo: datasets, checkpoints, prediction files and the paper PDF. The notebooks download or regenerate them.

Every number in the critique and the slides traces to a CSV or log in this repo, or to a table in the paper.
