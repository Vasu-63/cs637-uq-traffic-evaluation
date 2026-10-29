"""Evaluate one of the authors' shipped checkpoints and save val + test predictions.

Run from INSIDE a variant folder of the authors' repo, e.g.
    cd .../uq4phy_traffic_network/gpde_quantile_phy_model_PEMSD4
    python /path/eval_checkpoint.py --data_dir /content/pemsd4 --out /path/model.npz
The authors' code is not modified: we load their pretrained config with yaml.safe_load
(their run_demo uses yaml.load(f), which fails on PyYAML 6) and let torch.load read
their full checkpoint dict (PyTorch >= 2.6 defaults to weights_only=True).
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch, yaml

_torch_load = torch.load
def _load(*a, **k):
    k.setdefault('weights_only', False)
    return _torch_load(*a, **k)
torch.load = _load

sys.path.insert(0, os.getcwd())
from lib.utils import load_graph_data                      # noqa: E402  (authors' code)
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor  # noqa: E402
import inspect, textwrap                                    # noqa: E402
import model.pytorch.dcrnn_supervisor as _sup               # noqa: E402

# In the PEMSD8 variants evaluate() builds the predictions but its return line ends in
# "loss_mae#,{'prediction': ...}" (commented out). Re-enable it here, without editing their file.
_src = inspect.getsource(DCRNNSupervisor.evaluate)
if "#,{'prediction'" in _src:
    _ns = {}
    exec(textwrap.dedent(_src.replace("#,{'prediction'", ",{'prediction'")), _sup.__dict__, _ns)
    DCRNNSupervisor.evaluate = _ns['evaluate']
    print('NOTE: re-enabled the prediction output of evaluate() (commented out in this variant)')

p = argparse.ArgumentParser()
p.add_argument('--data_dir', required=True)
p.add_argument('--out', required=True)
p.add_argument('--log_dir', default='/content/p1_eval_logs')
p.add_argument('--epoch', type=int, default=None, help='default: epoch in the pretrained config')
p.add_argument('--config', default=None, help='default: data/model/pretrained/*/config.yaml (shipped checkpoints)')
a = p.parse_args()

cfg_path = a.config or sorted(glob.glob('data/model/pretrained/*/config.yaml'))[0]
cfg = yaml.safe_load(open(cfg_path))
saved = sorted(int(f[3:-4]) for f in os.listdir('models') if f.startswith('epo') and f.endswith('.tar'))
epoch = a.epoch if a.epoch is not None else cfg['train']['epoch']
assert epoch in saved, f'epo{epoch}.tar not in models/ (have {saved})'
if epoch != max(saved):
    print(f'NOTE: config epoch {epoch} is not the latest saved checkpoint ({max(saved)})')
cfg['data']['dataset_dir'] = a.data_dir
cfg['train']['epoch'] = epoch
cfg['train']['log_dir'] = os.path.join(a.log_dir, os.path.basename(os.getcwd()))

_, _, adj = load_graph_data(cfg['data']['graph_pkl_filename'])
t0 = time.time()
sup = DCRNNSupervisor(adj_mx=adj, **cfg)
result, meta = {}, {'variant': os.path.basename(os.getcwd()), 'epoch': epoch, 'config': cfg_path,
                    'reg_lambda': cfg['model'].get('reg'), 'seed': os.environ.get('SEED'),
                    'test_batch_size': cfg['data'].get('test_batch_size'),
                    'device': str(next(sup.dcrnn_model.parameters()).device)}
for split in ['val', 'test']:
    mis, width, mse, rmse, mae, out = sup.evaluate(split)
    n = len(np.load(os.path.join(a.data_dir, split + '.npz'))['y'])
    pred = np.stack(out['prediction']).astype(np.float32)   # (12, N_padded, nodes, 3) = lower, median, upper
    truth = np.stack(out['truth']).astype(np.float32)       # (12, N_padded, nodes)
    near0 = (truth != 0) & (np.abs(truth) < 1e-3)           # missing readings come back as float residues
    meta[f'{split}_missing_share'] = float((np.abs(truth) < 1e-3).mean())
    truth[near0] = 0.0
    meta[f'{split}_samples'] = n
    meta[f'{split}_padded_samples'] = int(pred.shape[1])
    meta[f'{split}_authors_metrics'] = dict(mae=float(mae), mis=float(mis), width=float(width), mse=float(mse), rmse=float(rmse))
    result[f'{split}_pred'] = pred[:, :n]        # drop the copies of the last sample used to pad the final batch
    result[f'{split}_truth'] = truth[:, :n]
    print(f'{split}: authors-style (all 12 horizons, incl. padding)  MAE {mae:.4f}  MIS {mis:.4f}  '
          f'width {width:.4f}  MSE {mse:.4f}  RMSE {rmse:.4f}   [{n} samples, padded to {pred.shape[1]}]')
meta['seconds'] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez_compressed(a.out, meta=json.dumps(meta), **result)
print('saved', a.out, f'({os.path.getsize(a.out)/1e6:.0f} MB, {meta["seconds"]} s)')
