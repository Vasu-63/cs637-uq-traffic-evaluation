# Critique

**Paper:** Bao, Nguyen, Liu, Ma, Johnson, Wei. *Uncertainty Quantification for Physics-Informed Traffic Graph Networks with Adversarial Attacks.* ACM Transactions on Cyber-Physical Systems, 2026. DOI 10.1145/3844727.

**Verdict.** The code runs and its numbers reproduce, but the paper's new contribution does not deliver what is claimed. Over five paired seeds, two of them run with the authors' own physics code, the Underwood physics loss never improves accuracy on average, and it makes 60-minute intervals slightly worse in every seed. The accuracy gains come from the GPDE layer, which is the authors' earlier work. The prediction intervals are not calibrated, and the robustness advantage under attack comes from testing the physics models against a weaker attack. The tables also report short-range averages as long-range errors, and single runs as means of five.

Numbers in brackets, such as (41), refer to the team's numbered findings. Every number below traces to a file in this repo or to a table in the paper.

## Claims against evidence

| Paper's claim | What we found | Evidence |
|---|---|---|
| Up to 14.8% lower short-term error, 8.7% long-term | The 14.8% is real but almost all GPDE (13.4 of the 14.8 points in the paper's own table). In our runs the physics loss adds −1.2 points over 3 seeds, i.e. it is slightly worse, and over all 5 seeds its 15-minute MAE is 1.9% higher than without it (1.046 → 1.066) (41, 42) | `reproduction/results/pemsd8_seed_check.csv`, `reproduction/results/rerun/` |
| The physics loss improves accuracy | No gain on PEMSD8 over 5 paired seeds, 2 of them with the authors' own physics code: physics is better in only 1 of 5 seeds at each horizon. No gain on METR-LA over 2 seeds. The paper's gain, 0.017 MAE, is smaller than our seed spread of 0.016–0.028 (27, 28, 41) | `reproduction/results/rerun/`, `implementation/results/metrla_tod_summary.csv` |
| Better-calibrated intervals | Nominal 95% intervals cover 45–65% of true speeds at 60 min, and 14–50% in congested traffic. Coverage is never measured in the paper (9, 30) | `extensions/results/coverage_conformal_metrics.csv` |
| More robust under adversarial attack | The physics models were attacked with a weaker PGD than their baselines. Under the same attack, their advantage equals their clean-accuracy gap: they are not more robust (44) | `analysis/table3_reattack_summary.csv` |
| Results are means of five runs | Table 1's PEMSD4 rows and 16 of 18 Table 3 Quantile rows equal single logged runs to 3–4 decimals (16, 43) | `reproduction/results/pemsd4_table1_vs_authors_logs.csv`, `analysis/table3_vs_logs.csv` |

## What holds up

**The code is released and the numbers reproduce.**
- Evaluating the six shipped PEMSD4 checkpoints gives the authors' logged test metrics to 3–4 decimals (38).
- Retrained from scratch on PEMSD8, GPDE+Quantile matches Table 1 within 0.1–0.3% on MAE and RMSE (40).
- GPDE+Quantile+Phy is within 2–4%, and plain Quantile within 3–6%. That is a normal gap for retrained networks.

So the paper's numbers are genuine outputs of its code. They are measured in the cumulative-average reading described below.

**The method is clear enough to rebuild.** Our own physics-loss module, written from the paper and code, matches the authors' loss and its gradients to 1e-5 on identical inputs (21, `implementation/tests/`). On PEMSD8 the per-sensor Underwood parameters are fitted on the training split only, as they should be (20).

**The GPDE layer gives large, real gains.**
- It cuts PEMSD8 15-minute MAE by about 18% against plain Quantile in our runs (42).
- It improves interval scores that survive calibration (31).
- It keeps a failed sensor from corrupting its neighbours. With 40% of sensors dead, healthy-sensor MAE rises from 2.72 to 6.72 mph in the plain model, but only from 2.19 to 2.50 with GPDE (32).

This is prior work by the same authors, but the paper shows it combines well with uncertainty methods.

**The model is cheap enough to deploy.**
- One network-wide forecast with intervals takes 49–58 ms on a single CPU core, under 0.02% of the 5-minute sensor period (33).
- About 25,000 parameters, 0.1 MB.
- The physics loss adds nothing at inference.

**The question matters.** Prediction intervals are what a traffic controller needs to act on a forecast, and the paper puts uncertainty, physics and attacks in one study. All three are central to cyber-physical systems.

## What does not hold up

### The physics loss gives no measurable gain

| Test | Result | Findings |
|---|---|---|
| PEMSD8, 5 paired seeds: GPDE+Quantile vs +Phy (seeds 0–2 with our re-implemented loss, seeds 3–4 with the authors' own code) | MAE with physics is 0.9–1.6% higher. Physics is better in only 1 of 5 seeds at 15, 30 and 60 min, and every gap is under one seed standard deviation | 41, `reproduction/results/rerun/` |
| PEMSD8, intervals at 60 min, same 5 pairs | Worse with physics in all 5 seeds: MIS +2.5% (paired t-test p = 0.03) and coverage −3.1 points (p = 0.009). The only consistent effect of the physics loss is slightly worse long-range intervals | `reproduction/results/rerun/` |
| Authors' own physics code vs our re-implementation | Same behaviour: single-horizon MAE ranges overlap at every horizon (15 min: 1.235–1.290 vs 1.234–1.268) | 21, `reproduction/results/rerun/` |
| Underwood vs the simpler Greenshields law, 3 seeds each | Indistinguishable. Paired differences change sign across seeds for every metric. Greenshields fits the training data better on 162 of 170 sensors | 22, 27 |
| METR-LA, 2 paired seeds | MAE equal or slightly worse with physics in all 6 seed × horizon pairs | 28 |
| PEMSD4 interval score, after conformal calibration | The physics models' 10–12% raw MIS advantage shrinks to within ±3%. For MAEMIS, +Phy is worse in every calibrated case | 31 |

PEMSD4's shipped single checkpoints do show a 5–7% MAE gain for Quantile with physics. That is one run per model, so it cannot be told apart from seed noise. On PEMSD8, where we could test across seeds, the gain disappears.

### The tables overstate accuracy

- **"15 / 30 / 60 min" are cumulative averages (15, 35, 39, 40).** Table 1 reports the average error over all steps up to each horizon, not the error at that horizon. On PEMSD8, the true 60-minute MAE is about 2.0 (2.017), where Table 1 shows 1.52, so the table understates it by about a quarter. It understates the 60-minute interval score (46.8 vs 28.8) by about 40%.
- **Duplicated test samples are counted (39).** The evaluation pads its last batch with 208 copies of the final PEMSD4 test sample, and those copies enter the reported metrics. Adding them back brings our numbers to within 0.03–0.2% of Table 1. Without them, the gap is 2–7%.
- **Single runs, no error bars (9, 16, 43).** Section 6.1.2 says each experiment was run five times and the mean reported. In fact:
  - the PEMSD4 rows of Table 1 equal one logged run each
  - one model has four logged runs, and Table 1 matches the last
  - 16 of 18 Quantile rows in Table 3 also equal a single logged run
  - the random seed is fixed at 0 in the code
  - no standard deviations are reported

  Our results show why this matters. Seed 0 alone made Greenshields look clearly better than Underwood, and seed 1 reversed it (27).

### The intervals are not calibrated

- **Coverage collapses with horizon (30).** The nominal 95% intervals cover 95% only at the first 5-minute step. At 60 minutes they cover 52–65% on PEMSD4 (Quantile trio) and 58–61% on PEMSD8 (GPDE models). The intervals get narrower as the horizon grows, when they should widen.
- **Congestion is worst.** Below 40 mph, where a controller most needs a warning, coverage is 14–50%.
- **Conformal calibration fixes it, at a price.** It restores 95% coverage on the test set, with intervals 3.2–4.6 times wider (30).

The paper calls its intervals "calibrated" but never measures coverage.

### The robustness claim depends on an unequal attack

- **Different attacks for different models (44).** In the PGD rows of Table 3:
  - the baselines are attacked from the clean input
  - the physics models are attacked from a random start
  - no config overrides either setting
- **The random start is weaker.** Re-running the authors' own attack code on their shipped checkpoints confirms it: MAE is 1.4–1.8% lower for every model.
- **Under the same attack, the advantage disappears.** The physics models' edge becomes −4.4% on PEMSD8 (Table 3: −5.7%) and −2.5% on METR-LA (Table 3: −4.2%). That is the same size as their edge with no attack at all. The error increase from clean to attacked is almost identical with and without physics, for example +6.0% vs +5.7% on PEMSD8.

The physics models are slightly more accurate, but not more robust.

- **Two Table 3 rows have no shipped log (43).** The two PEMSD4 GPDE+Quantile+Phy rows of Table 3 match no shipped log, and there is no PEMSD4 PGD folder for that model at all. Those are exactly the rows where physics was worse than GPDE alone (4).
- **The threat model is narrow (10).** It uses a 2-step PGD and gradient attacks only. It includes no realistic faults such as dead or stuck sensors; we tested those instead (32).

### Code problems behind the results

- **The physics law is applied 60 minutes out of step (18).** The loss compares each predicted speed with the occupancy from 12 steps (60 minutes) earlier, at every horizon. The Underwood law relates speed and density at the same time and place. Fixing the alignment gives modestly but consistently better intervals: MIS 3–6% lower in all 9 seed × horizon pairs. The gain is largest in congestion, where it survives calibration (36, 37).
- **On METR-LA, "density" is the time of day (7, 23, 24).** METR-LA has no occupancy, and the feature the physics loss reads is the clock. Its fitted parameters are unphysical: k_m is negative on 147 of 207 sensors, and the physics target sits below each sensor's mean speed on all 207 sensors, by about 20 mph. This explains why physics does nothing there.
- **The diffusion matrices are scrambled (14).** `DCGRUCell._build_sparse_matrix`, inherited from DCRNN-PyTorch, re-sorts the sparse indices but not their values. On the PEMSD4 graph, the stored matrix differs from the intended one by up to 0.70, where the largest entry is 1.0. Every reported and reproduced result was trained this way; the effect on accuracy is untested.
- **Smaller issues:**
  - The GPDE layer's hard-coded constants were fitted on data that includes the test period (20).
  - The physics loss hard-codes PEMSD8's shape and batch size (19).
  - The reported "MIS" adds the MAE to the interval score of Eq. 4 (13).
  - Eq. 9 omits the square the code applies (8).
  - The PEMSD8 folders cannot reload their own checkpoints, and their evaluation output is commented out (`reproduction/README.md`).
  - The PEMSD4 processing script shipped in the repo is the PEMSD8 one (`reproduction/README.md`).

### Errors and omissions in the text

- Table 1's PEMSD4 RMSE column is not the square root of its MSE column in any of 27 rows. Table 2 gives the correct values for the same run (1).
- PEMSD4 MAEMIS 60-minute MAE is printed as 3.167; the matching log gives 2.6805 (17).
- One EDL RMSE is a typo: 2.698 against sqrt(MSE) = 2.598 (6).
- The λ values behind Table 1 are never stated: 0.005 for PEMSD4 and 0.002 for PEMSD8, found by matching Table 2 (2, 34). The METR-LA value is 0.0005 (24).
- Section 6.1.2 says "two datasets" when there are three (12).
- Fig. 15's caption names the same location for both panels (12).

## What we added

| Area | Addition | Folder |
|---|---|---|
| Implementation | Our own physics-loss module with 11 tests, swapped into the authors' model and retrained | `implementation/` |
| Reproduction | PEMSD4 checkpoint evaluation; PEMSD8 retraining; per-horizon comparison in both readings; 3-seed check of the physics loss | `reproduction/` |
| New ideas | Greenshields vs Underwood; a time-aligned physics loss (fix for 18); conformal calibration of the intervals (CQR, per-sensor CQR, ACI) | `implementation/`, `extensions/` |
| New use cases | Dead and stuck sensors, with a graph-neighbour repair; CPU latency and model size; Table 3 re-attack with the same PGD for every model | `extensions/`, `analysis/` |

## Through the course's lens

A traffic forecaster sits in a sense–compute–actuate loop: loop detectors sense, the model computes, and a signal controller or ramp meter acts. Seen that way:

- **Timing is not the problem.** A 58 ms forecast against a 5-minute sampling period leaves the computation far inside its deadline, even on one CPU core (33).
- **The interval is the safety signal, and it is the weak link.** An operator who trusts a "95%" interval that covers 14–50% of congested cases acts on false confidence when the stakes are highest. Conformal calibration is a cheap, model-agnostic fix and should be part of any deployment.
- **Real faults matter more than gradient attacks.** Dead sensors are a routine failure in road networks. They spread errors through the graph unless the architecture (GPDE) or an explicit repair contains them (32). The paper's robustness study does not cover this case.
- **Verification is missing.** The paper offers no guarantee on forecasts or intervals, and lists formal verification only as future work (11). For a safety-relevant CPS component, calibrated coverage guarantees, which conformal methods provide on exchangeable data, are a practical first step.

## Verdict

The paper asks the right question: can physics knowledge make traffic forecasts both more accurate and more trustworthy? Its released code is good enough that we could reproduce its numbers closely. But the evidence for its own contribution does not survive a careful re-run:
- across five paired seeds, including two with the authors' own code, the Underwood loss never helps on average and makes 60-minute intervals slightly worse in every seed
- the accuracy gains belong to the GPDE layer from the authors' earlier work
- the "calibrated" intervals cover as little as half of true speeds at 60 minutes
- the robustness advantage comes from attacking the physics models more gently

The tables compound this: averages are reported as long-range errors, padded duplicates are counted, and single runs are reported as means. Our own fix for the physics loss's 60-minute misalignment and conformal calibration both help. That suggests the idea is worth pursuing, with time-aligned physics, calibrated intervals, multiple seeds and per-horizon reporting.
