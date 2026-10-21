# extensions/: calibration, sensor failures, latency

Three experiments the paper does not do, each tied to a critique point:

| Script | Question | Critique point |
|---|---|---|
| `coverage_conformal.py` | Do the "95%" intervals contain the truth 95% of the time? Can conformal calibration fix them? | 9: "calibrated" is claimed but coverage is never measured |
| `sensor_failure.py` | What happens when loop detectors go dead or stuck, and does a simple detect-and-impute step help? | 10: gradient attacks only, no realistic faults |
| `latency.py` | How fast and how big is the model on a CPU, against the 5-minute sampling period? | 11: no timing, memory or deployment analysis |

`p3_model_utils.py` loads the authors' checkpoints outside their training code (any batch size, CPU or GPU). `P3_01_calibration_failure_latency.ipynb` runs everything on Colab.

## What each script does

**Coverage + conformal.** Reads Person 1's prediction files. Reports per horizon: MAE, RMSE, coverage, mean width, standard MIS, the authors' "MIS" (standard MIS + MAE, finding 13), quantile-crossing rate, per-sensor coverage spread, and coverage split by regime (congested below 40 mph, 40 to 60, free flow). Then recalibrates the intervals with the validation split as calibration set and scores them on test:

- `cqr`: split conformalized quantile regression (Romano et al., NeurIPS 2019), one correction per horizon.
- `cqr_sensor`: the same, one correction per horizon and sensor, so each sensor gets its own guarantee.
- `aci`: adaptive conformal inference (Gibbs and Candès, NeurIPS 2021), online over the test period per horizon and sensor. Feedback is delayed by the horizon, since the truth of a 60-minute forecast is known only 60 minutes later. Where textbook ACI would output an infinite interval we cap at the largest calibration score, and report how often that happens.

Caveat for the slides: split conformal assumes exchangeable data. Validation (weeks before test) and test are a time series, so CQR's guarantee is only approximate; ACI is the variant designed for that.

Self-test (`--selftest`, synthetic data, no files needed): raw intervals 40% too narrow cover 77%; all three methods restore 95%; per-sensor CQR fixes per-sensor coverage (5th percentile sensor 0.84 to 0.95); under a shift (test 30% noisier than validation) static CQR drops to 0.89 and ACI holds 0.95.

**Sensor failures.** Breaks 5/10/20/40% of the 307 PEMSD4 sensors for the whole test period, in the model input only (truth untouched), over 3 random sensor sets. `dead`: all channels read 0, as PeMS stores missing data. `stuck`: the reading freezes at the first step of each hour-long input window. Fix: flag a sensor-window as faulty when it is all zeros or flat in every channel, replace its input with the weighted mean of healthy graph neighbours (fallback: training mean). Reports metrics on all sensors, the failed ones, and healthy direct neighbours of failed ones, plus the detector's false-alarm rate on clean data and its recall.

**Latency.** Parameter count (and the GPDE layer's own share), checkpoint size, latency of one network-wide forecast at batch 1/16/64 with 1 thread and all threads. CPU only. `peak_rss_MB` is the whole process (Python, PyTorch and the loader's batch-256 warm-up pass), not the model's own memory.

## Run order

1. Put the four `.py` files in `MyDrive/CS637/p3/extensions/` and open the notebook in Colab.
2. Section 2 latency (CPU is enough, about 3 min).
3. Section 3 sensor failures (T4 GPU). The first cell builds PEMSD4 with Person 1's `p1/scripts/process_pems.py` from `data/raw/pems04.npz`. Then the `--clean_only` check: clean test MAE should be within a few percent of the authors' log at the saved epoch (quantile 2.6702, +GPDE 2.1338, +GPDE+Phy 2.0189; `<variant>/data/model/dcrnn_*/info.log`). Then the full run (roughly 20 to 40 min).
4. Section 4 coverage: on Person 1's `outputs/pemsd4/*.npz` (22 Oct), and on PEMSD8 when those come (24 Oct).

## Checks done so far (21 Oct)

- Loader: clean MAE, interval width and the authors' MIS are within 1 to 3% of the authors' logs for all three models, ours always slightly higher. Expected cause: the authors' evaluation pads the test set from 3376 to 3584 with copies of the last sample and averages per batch; we score the 3376 real samples only. Width, which does not depend on the truth, shows the same small offset.
- Latency: two independent runs agree at batch 1 (49.3 vs 50.6 ms for the plain Quantile model, 55.4 vs 55.1 ms with GPDE).

## Things to watch in the results

- Check coverage in the congested regime separately: that is when an operator needs the interval.
- Whether +Phy helps under sensor faults is an open question; the Underwood term reads occupancy, which is exactly what a faulty sensor corrupts.
- The detect-and-impute fix also replaces stuck (flat) sensors with their neighbours' average. Early results suggest that hurts compared with leaving a stuck sensor's stale value in place; compare `fix` rows for `stuck` before recommending it.

## Results

Results CSVs and figures are added after the 25 Oct results freeze; until then they live in `MyDrive/CS637/p3/`.
