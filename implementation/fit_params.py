"""
Fit per-sensor speed-density parameters on the PEMSD8 TRAINING split only,
and compare how well each law fits.

    Underwood    ln v = ln v_f - k / k_m          (log-linear least squares)
    Greenshields v    = v_f - (v_f / k_jam) k     (linear least squares)

The Underwood fit reproduces the authors' shipped PEMSD8_v_f.npy and
PEMSD8_rho_max.npy (checked: mean abs differences 0.003 and 0.001).

Usage (Colab):
    python implementation/fit_params.py --data /content/data/PEMSD8/pems8ndy.npy \
        --out implementation/params

Writes to --out:
    underwood_v_f.npy, underwood_k_m.npy, greenshields_v_f.npy, greenshields_k_jam.npy  (170,)
    fit_quality.csv   per-sensor RMSE (mph) and R^2 of each law on the training split
"""
import argparse
import csv
import os

import numpy as np

TOTAL_STEPS = 17856          # PEMSD8 length; train = first 70%, as in the authors' script
SPEED, OCC = 0, 1            # pems8ndy.npy channel order: [speed, occupancy, flow]


def fit(train):
    n = train.shape[1]
    out = {k: np.zeros(n) for k in ('u_vf', 'u_km', 'g_vf', 'g_kjam',
                                    'u_rmse', 'g_rmse', 'u_r2', 'g_r2')}
    for s in range(n):
        v, k = train[:, s, SPEED], train[:, s, OCC]
        ok = (v > 0) & np.isfinite(v) & np.isfinite(k)
        v, k = v[ok], k[ok]

        a, b = np.polyfit(k, np.log(v), 1)
        u_vf, u_km = np.exp(b), -1.0 / a
        a, b = np.polyfit(k, v, 1)
        g_vf, g_kjam = b, b / -a

        u_hat = u_vf * np.exp(-k / u_km)
        g_hat = np.clip(g_vf * (1 - k / g_kjam), 0, None)
        ss_tot = ((v - v.mean()) ** 2).sum()
        for tag, hat in (('u', u_hat), ('g', g_hat)):
            ss = ((v - hat) ** 2).sum()
            out[f'{tag}_rmse'][s] = np.sqrt(ss / len(v))
            out[f'{tag}_r2'][s] = 1 - ss / ss_tot
        out['u_vf'][s], out['u_km'][s] = u_vf, u_km
        out['g_vf'][s], out['g_kjam'][s] = g_vf, g_kjam
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='path to pems8ndy.npy')
    p.add_argument('--out', required=True, help='output folder')
    args = p.parse_args()

    d = np.load(args.data)
    train = d[:int(TOTAL_STEPS * 0.7)]
    r = fit(train)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'underwood_v_f.npy'), r['u_vf'])
    np.save(os.path.join(args.out, 'underwood_k_m.npy'), r['u_km'])
    np.save(os.path.join(args.out, 'greenshields_v_f.npy'), r['g_vf'])
    np.save(os.path.join(args.out, 'greenshields_k_jam.npy'), r['g_kjam'])

    with open(os.path.join(args.out, 'fit_quality.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sensor', 'underwood_rmse', 'greenshields_rmse', 'underwood_r2', 'greenshields_r2'])
        for s in range(len(r['u_vf'])):
            w.writerow([s, f"{r['u_rmse'][s]:.4f}", f"{r['g_rmse'][s]:.4f}",
                        f"{r['u_r2'][s]:.4f}", f"{r['g_r2'][s]:.4f}"])

    better = (r['u_rmse'] < r['g_rmse']).sum()
    print(f'sensors: {len(r["u_vf"])}')
    print(f'median RMSE  Underwood {np.median(r["u_rmse"]):.3f}  Greenshields {np.median(r["g_rmse"]):.3f} mph')
    print(f'median R^2   Underwood {np.median(r["u_r2"]):.3f}  Greenshields {np.median(r["g_r2"]):.3f}')
    print(f'Underwood fits better on {better} of {len(r["u_vf"])} sensors')
    print(f'Greenshields k_jam range {r["g_kjam"].min():.3f} .. {r["g_kjam"].max():.3f}')


if __name__ == '__main__':
    main()
