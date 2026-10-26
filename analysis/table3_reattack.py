"""
Re-attack the authors' adversarially trained PGD checkpoints (Table 3) with the same PGD for every model.

In the authors' code the baselines' test-time PGD starts from the clean input (evaluate_pgd default
random_start=False), while the physics models' PGD starts from a random point (pgd_random_eval=True).
This script evaluates each shipped checkpoint with both settings, using the authors' own evaluate_pgd
(eps = 0.03, 2 steps, step 0.015), plus a clean evaluation, on the test split.

Run in Colab (GPU optional). It fetches only the four needed folders of
https://github.com/tianshubao1/uq4phy_traffic_network and the PEMSD8 / METR-LA data from figshare.
Output: table3_reattack.csv (one row per model x setting x seed).
"""
import glob, json, os, shutil, subprocess, requests

FOLDERS = [('gpde_quantile_model_PEMSD8_PGD', 18, 'PEMSD8'),           # Table 3 PGD PEMSD8 GPDE+Quantile
           ('gpde_quantile_phy_model_PEMSD8_PGD_train', 18, 'PEMSD8'),  # Table 3 PGD PEMSD8 GPDE+Quantile+Phy
           ('gpde_quantile_model_PGD', 16, 'METR-LA'),                  # Table 3 PGD METR-LA GPDE+Quantile
           ('gpde_quantile_phy_model_PGD_train', 19, 'METR-LA')]        # Table 3 PGD METR-LA GPDE+Quantile+Phy
FIGSHARE = {'PEMSD8': 28347662, 'METR-LA': 28347830}
ROOT = '/content'

RUNNER = r'''
import sys, os, json, yaml, torch, numpy as np
folder, epoch, data_dir, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
os.chdir(folder); sys.path.insert(0, folder)
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})
from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
cfg = yaml.safe_load(open('data/model/dcrnn_la.yaml'))
cfg['data']['dataset_dir'] = data_dir
cfg['train']['epoch'] = epoch
cfg['train']['log_dir'] = os.path.join(os.path.dirname(out), 'attack_logs', os.path.basename(folder))
_, _, adj = load_graph_data(cfg['data']['graph_pkl_filename'])
sup = DCRNNSupervisor(adj_mx=adj, **cfg)
rows = []
for setting, eps, rs, seed in [('clean', 0.0, False, 0), ('pgd_fixed_start', 0.03, False, 0),
                               ('pgd_random_start', 0.03, True, 0), ('pgd_random_start', 0.03, True, 1),
                               ('pgd_random_start', 0.03, True, 2)]:
    torch.manual_seed(seed); np.random.seed(seed)
    mis, width, mse, rmse, mae, _ = sup.evaluate_pgd('test', epsilon=eps, steps=2, random_start=rs)
    rows.append(dict(model=os.path.basename(folder), epoch=epoch, setting=setting, seed=seed,
                     mae=mae, mis=mis, mse=mse, rmse=rmse, width=width))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(out, 'w'))
'''


def main():
    repo = f'{ROOT}/authors'
    if not os.path.isdir(repo):
        subprocess.run(f'git clone -q --depth 1 --filter=blob:none --no-checkout '
                       f'https://github.com/tianshubao1/uq4phy_traffic_network {repo}', shell=True, check=True)
        subprocess.run(['git', '-C', repo, 'checkout', 'HEAD', '--'] + [f for f, _, _ in FOLDERS], check=True)
    for p in glob.glob(f'{repo}/**/__pycache__', recursive=True):
        shutil.rmtree(p)
    for name, aid in FIGSHARE.items():
        out = f'{ROOT}/data/{name}'
        if not os.path.isdir(out):
            os.makedirs(out)
            for f in requests.get(f'https://api.figshare.com/v2/articles/{aid}/files').json():
                with requests.get(f['download_url'], stream=True) as r, open(os.path.join(out, f['name']), 'wb') as fh:
                    for c in r.iter_content(1 << 20):
                        fh.write(c)
    open(f'{ROOT}/attack_runner.py', 'w').write(RUNNER)
    rows = []
    for folder, ep, ds in FOLDERS:
        out = f'{ROOT}/attack_{folder}.json'
        subprocess.run(['python', f'{ROOT}/attack_runner.py', f'{repo}/{folder}', str(ep), f'{ROOT}/data/{ds}', out], check=True)
        rows += json.load(open(out))
    import pandas as pd
    pd.DataFrame(rows).to_csv(f'{ROOT}/table3_reattack.csv', index=False)


if __name__ == '__main__':
    main()
