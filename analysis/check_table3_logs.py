"""
Trace the Quantile rows of Table 3 (adversarially trained models, FGSM and PGD)
to the training logs shipped in the authors' repository.

Usage: run from the root of a checkout of tianshubao1/uq4phy_traffic_network
(only the info.log files of the *_PGD / *_FGSM* Quantile folders are needed):
    python check_table3_logs.py > table3_vs_logs.csv
A row matches when MAE and at least one of MIS / MSE agree within 0.0011
(Table 3 prints 3 decimals; the logs print 4, and a few table entries are rounded up by one unit).
"""
import glob, re

TABLE3 = {  # (attack, dataset): {model: (MAE, MIS, MSE)}, Quantile rows of Table 3
    ('PGD', 'METR-LA'): {'Quantile': (4.084, 67.483, 64.411), 'GPDE+Quantile': (4.069, 65.815, 61.453), 'GPDE+Quantile+Phy': (3.898, 61.998, 59.531)},
    ('PGD', 'PEMSD8'):  {'Quantile': (2.447, 53.912, 23.928), 'GPDE+Quantile': (1.795, 33.123, 15.340), 'GPDE+Quantile+Phy': (1.693, 32.510, 13.783)},
    ('PGD', 'PEMSD4'):  {'Quantile': (2.889, 62.040, 35.323), 'GPDE+Quantile': (2.176, 42.220, 21.203), 'GPDE+Quantile+Phy': (2.377, 46.822, 23.785)},
    ('FGSM', 'METR-LA'): {'Quantile': (4.055, 65.613, 62.532), 'GPDE+Quantile': (3.992, 62.665, 61.237), 'GPDE+Quantile+Phy': (3.972, 62.937, 62.368)},
    ('FGSM', 'PEMSD8'):  {'Quantile': (2.072, 43.617, 18.083), 'GPDE+Quantile': (2.031, 41.761, 17.929), 'GPDE+Quantile+Phy': (2.046, 43.158, 17.578)},
    ('FGSM', 'PEMSD4'):  {'Quantile': (3.153, 68.328, 38.514), 'GPDE+Quantile': (2.068, 43.469, 19.990), 'GPDE+Quantile+Phy': (2.347, 45.972, 22.641)},
}
RX = re.compile(r'Epoch \[(\d+)/\d+\]\s*\(\d+\)\s*(\[[^\]]*\])?\s*test_mae_adv:\s*([\d.]+),\s*test_mis_adv:\s*([\d.]+),.*?test_mse_adv:\s*([\d.]+)')

lines = []
for f in sorted(glob.glob('*/data/model/**/info.log', recursive=True)):
    for l in open(f, errors='ignore'):
        m = RX.search(l)
        if m:
            lines.append((f, int(m[1]), (m[2] or '').strip('[]'), float(m[3]), float(m[4]), float(m[5])))

print('attack,dataset,model,table_mae,table_mis,table_mse,log_file,log_epoch,log_attack_tag,log_mae,log_mis,log_mse')
for (att, ds), rows in TABLE3.items():
    for model, (mae, mis, mse) in rows.items():
        hits = [x for x in lines if abs(x[3] - mae) < 0.0011 and (abs(x[4] - mis) < 0.0011 or abs(x[5] - mse) < 0.0011)]
        if not hits:
            print(f'{att},{ds},{model},{mae},{mis},{mse},NO MATCHING LOG LINE,,,,,')
        for f, ep, tag, a, b, c in hits:
            print(f'{att},{ds},{model},{mae},{mis},{mse},{f},{ep},"{tag}",{a},{b},{c}')
