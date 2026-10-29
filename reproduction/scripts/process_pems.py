"""Build train/val/test.npz for the authors' DCRNN code from a raw PEMS array.

Faithful to the authors' 'PEMSD8 data processing.py' (70/10/20 split by time,
12-step input window, the next 12 steps as target, channels [speed, occupancy, flow]),
but vectorised (their loop writes x[i:] instead of x[i], which is slow but gives
the same final arrays) and with sizes taken from the data instead of hard-coded.
"""
import argparse, os, numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def windows(seg, T=12):
    n = len(seg) - 2 * T + 1          # authors: len - 23 samples
    w = sliding_window_view(seg, 2 * T, axis=0)      # (n, nodes, ch, 24)
    w = np.moveaxis(w, -1, 1)[:n]                    # (n, 24, nodes, ch)
    return np.ascontiguousarray(w[:, :T]), np.ascontiguousarray(w[:, T:])

def main(a):
    raw = np.load(a.raw)
    data = raw[a.key] if hasattr(raw, 'files') else raw
    data = data[..., [int(c) for c in a.order.split(',')]].astype(np.float64)
    N = len(data); tr, va = int(N * 0.7), int(N * 0.8)
    os.makedirs(a.out, exist_ok=True)
    for name, seg in [('train', data[:tr]), ('val', data[tr:va]), ('test', data[va:])]:
        x, y = windows(seg)
        np.savez(os.path.join(a.out, name + '.npz'), x=x, y=y)
        print(f'{name}: x {x.shape} y {y.shape}')
    print('channel means [speed, occupancy, flow]:', data.reshape(-1, 3).mean(0).round(3))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--key', default='data')
    p.add_argument('--order', default='2,1,0', help='raw channel indices giving [speed, occupancy, flow]; PEMS04.npz is [flow, occupancy, speed]')
    main(p.parse_args())
