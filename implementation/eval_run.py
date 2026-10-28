"""Per-horizon evaluation of one of OUR training runs (copies of the authors' variant folders).

Based on the reproduction folder's checkpoint-evaluation script, with three changes:
  * uses the run's own training config data/model/dcrnn_la.yaml (not the authors' pretrained one);
  * loads models/epo<N>.tar from the run folder (the authors' load_model hard-codes a
    'gpde_quantile_phy_model_PEMSD8/models/...' path, so it is replaced at runtime);
  * computes 15/30/60-min metrics with horizon_metrics.py (reproduction scripts) and appends
    them to one CSV, so every run is scored the same way.
The authors' code and our run folders are not modified.

Usage (Colab), from anywhere:
    python implementation/eval_run.py --run <run folder> --epoch 30 --data_dir /content/data/PEMSD8 \
        --label pemsd8_underwood_ours --csv implementation/results/horizon_metrics.csv
For a Greenshields run also pass --law greenshields --vf <...>/greenshields_v_f.npy --k <...>/greenshields_k_jam.npy
(only the import needs them; evaluation itself does not use the physics loss).
"""
import argparse, csv, json, os, sys, time
import numpy as np, torch, yaml

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, help='run folder, e.g. runs/pemsd8_underwood_ours')
p.add_argument('--epoch', type=int, required=True, help='best epoch (last "saving to models/epoN.tar" in the log)')
p.add_argument('--data_dir', required=True)
p.add_argument('--label', required=True, help='name of this run in the CSV')
p.add_argument('--csv', required=True)
p.add_argument('--metrics_scripts', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reproduction', 'scripts'),
               help='folder containing horizon_metrics.py')
p.add_argument('--save_npz', default=None, help='optional path to save val/test predictions')
p.add_argument('--law', default='underwood')
p.add_argument('--vf', default=None)
p.add_argument('--k', default=None)
a = p.parse_args()

os.environ['PHYS_LAW'] = a.law
if a.vf: os.environ['PHYS_VF'] = a.vf
if a.k:  os.environ['PHYS_K'] = a.k

_torch_load = torch.load
def _load(*args, **kw):
    kw.setdefault('weights_only', False)
    return _torch_load(*args, **kw)
torch.load = _load

run = os.path.abspath(a.run)
os.chdir(run)
sys.path.insert(0, run)
sys.path.insert(0, os.path.abspath(a.metrics_scripts))
from lib.utils import load_graph_data                      # noqa: E402  (authors' code, in the run folder)
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor  # noqa: E402
from horizon_metrics import horizon_table                   # noqa: E402  (reproduction scripts)

ckpt = os.path.join(run, 'models', f'epo{a.epoch}.tar')
assert os.path.exists(ckpt), f'{ckpt} not found; have {sorted(os.listdir(os.path.join(run, "models")))}'

def _load_model(self):
    self._setup_graph()
    state = torch.load(ckpt, map_location='cpu')
    self.dcrnn_model.load_state_dict(state['model_state_dict'])
    self._logger.info(f'Loaded {ckpt}')
DCRNNSupervisor.load_model = _load_model

cfg = yaml.safe_load(open(os.path.join(run, 'data', 'model', 'dcrnn_la.yaml')))
cfg['data']['dataset_dir'] = a.data_dir
cfg['train']['epoch'] = a.epoch                      # epoch > 0 makes the supervisor call load_model
cfg['train']['log_dir'] = f'/content/eval_logs/{a.label}'

_, _, adj = load_graph_data(cfg['data']['graph_pkl_filename'])
t0 = time.time()
sup = DCRNNSupervisor(adj_mx=adj, **cfg)

# Some variants comment out the predictions in evaluate()'s return; re-enable at runtime (authors' files untouched).
import inspect, textwrap, sys as _sys
_S = _sys.modules[DCRNNSupervisor.__module__]
_src = textwrap.dedent(inspect.getsource(DCRNNSupervisor.evaluate))
if "loss_mae#,{" in _src:
    assert _src.count("y_preds_scaled") >= 2, "predictions are not built in evaluate(); cannot re-enable"
    _ns = {}
    exec(_src.replace("loss_mae#,{", "loss_mae,{"), _S.__dict__, _ns)
    DCRNNSupervisor.evaluate = _ns["evaluate"]
    print("evaluate(): prediction output re-enabled")

rows, saved = [], {}
meta = dict(label=a.label, run=run, epoch=a.epoch, law=a.law, reg_lambda=cfg['model'].get('reg'))
for split in ['val', 'test']:
    mis, width, mse, rmse, mae, out = sup.evaluate(split)
    n = len(np.load(os.path.join(a.data_dir, split + '.npz'))['y'])
    pred = np.stack(out['prediction']).astype(np.float32)[:, :n]   # drop padding copies of the last sample
    truth = np.stack(out['truth']).astype(np.float32)[:, :n]
    _near = (truth != 0) & (np.abs(truth) < 1e-3)
    print(f'{split}: truth exactly 0: {(truth == 0).mean()*100:.2f}%, float-residue zeros snapped to 0: {_near.mean()*100:.2f}%')
    truth[_near] = 0
    meta[f'{split}_authors_metrics'] = dict(mae=float(mae), mis=float(mis), width=float(width),
                                            mse=float(mse), rmse=float(rmse))
    print(f'{a.label} {split}: authors-style all-horizon (incl. padding) MAE {mae:.4f} RMSE {rmse:.4f} MIS {mis:.4f}')
    for r in horizon_table(pred, truth):
        rows.append(dict(label=a.label, split=split, epoch=a.epoch, law=a.law,
                         reg_lambda=cfg['model'].get('reg'), **r))
    if a.save_npz:
        saved[f'{split}_pred'], saved[f'{split}_truth'] = pred, truth

os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
new = not os.path.exists(a.csv)
with open(a.csv, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if new: w.writeheader()
    for r in rows:
        w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})

print(f'\n{a.label}, test set (step = single horizon, avg = mean over horizons up to it):')
print(f'{"view":5} {"min":>4} {"MAE":>7} {"RMSE":>7} {"MIS":>8} {"MIS_std":>8} {"width":>7} {"cover":>6}')
for r in rows:
    if r['split'] == 'test':
        print(f'{r["view"]:5} {r["minutes"]:>4} {r["mae"]:7.3f} {r["rmse"]:7.3f} {r["mis_authors"]:8.3f} '
              f'{r["mis_std"]:8.3f} {r["width"]:7.3f} {r["coverage"]:6.3f}')
if a.save_npz:
    np.savez_compressed(a.save_npz, meta=json.dumps(meta), **saved)
    print('saved', a.save_npz)
print(f'appended {len(rows)} rows to {a.csv} ({time.time() - t0:.0f} s)')
