"""Per-horizon metrics from the prediction files written by eval_checkpoint.py.

Two readings of "15 / 30 / 60 min", because the paper does not say which it uses:
  step : the single forecast step at that lead time (step 3, 6, 12 of 12)
  avg  : the average over all steps up to that lead time (steps 1-3, 1-6, 1-12);
         avg at 60 min is the all-horizon average the authors' code prints
MIS is reported two ways:
  mis_authors : the code's mis_loss = (u - l) + |y - median| + 40*(y - u)+ + 40*(l - y)+
  mis_std     : the paper's Eq. 4 / standard interval score, without the |y - median| term
Readings where truth == 0 are excluded, as in the authors' masked losses; |truth| < 1e-3 counts
as 0 too, because missing readings come back from evaluate() as tiny float residues.
pad_to_batch: add copies of the last sample the way the authors' DataLoader pads the final
batch (their reported numbers include these copies); use it only to match their numbers.
"""
import numpy as np

LEADS = {15: 3, 30: 6, 60: 12}           # minutes -> number of 5-minute steps
RHO = 2.0 / 0.05                         # 2 / alpha for a 95% interval

def _metrics(pred, truth):
    m = np.abs(truth) >= 1e-3
    lo, mid, hi, y = pred[..., 0][m], pred[..., 1][m], pred[..., 2][m], truth[m]
    lo, mid, hi, y = (v.astype(np.float64) for v in (lo, mid, hi, y))
    err = mid - y
    pen = RHO * np.maximum(y - hi, 0) + RHO * np.maximum(lo - y, 0)
    mse = np.mean(err ** 2)
    return dict(mae=np.mean(np.abs(err)), mse=mse, rmse=np.sqrt(mse),
                mis_authors=np.mean(hi - lo + np.abs(err) + pen), mis_std=np.mean(hi - lo + pen),
                width=np.mean(hi - lo), coverage=np.mean((y >= lo) & (y <= hi)),
                crossed=np.mean((lo > mid) | (mid > hi)))

def pad_like_authors(pred, truth, batch_size):
    n = pred.shape[1]; extra = (-n) % batch_size
    if extra == 0:
        return pred, truth
    return (np.concatenate([pred, np.repeat(pred[:, -1:], extra, axis=1)], axis=1),
            np.concatenate([truth, np.repeat(truth[:, -1:], extra, axis=1)], axis=1))

def horizon_table(pred, truth, pad_to_batch=None):
    """pred (12, N, nodes, 3), truth (12, N, nodes) -> list of dict rows."""
    if pad_to_batch:
        pred, truth = pad_like_authors(pred, truth, pad_to_batch)
    rows = []
    for minutes, k in LEADS.items():
        rows.append(dict(view='step', minutes=minutes, **_metrics(pred[k - 1], truth[k - 1])))
        rows.append(dict(view='avg', minutes=minutes, **_metrics(pred[:k], truth[:k])))
    return rows
