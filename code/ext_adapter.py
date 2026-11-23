# -*- coding: utf-8 -*-
"""把文献外的 IR 光谱数据集 (HuggingFace: Lamblador/IRSpectra) 适配为 ViBench 格式并落盘。"""
import sys, json, pickle, os
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

GRID = np.linspace(400, 4000, 1024)

def resample(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    yy = np.interp(GRID, x, y, left=0.0, right=0.0)
    yy = np.clip(yy, 0.0, None)
    return yy

def norm_variants(y):
    out = {}
    mx = y.max()
    out['max1'] = y / mx if mx > 0 else y
    s = y.sum()
    out['sum1'] = y / s if s > 0 else y
    n = np.linalg.norm(y)
    out['l2_1'] = y / n if n > 0 else y
    # ViBench 尺度: 逐点值约 0.01-0.3, 峰值约 1?
    out['raw'] = y
    return out

if __name__ == '__main__':
    t = pq.read_table('datasets/external_irspec/test.parquet')
    df = t.to_pandas()
    recs = []
    skipped = 0
    for _, r in df.iterrows():
        smi = str(r['smiles']).strip()
        if not smi:
            skipped += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue
        kek = Chem.MolToSmiles(mol, kekuleSmiles=True)
        x = r['spectrum_processed_x'] if r['spectrum_processed_x'] is not None else r['spectrum_raw_x']
        y = r['spectrum_processed_y'] if r['spectrum_processed_y'] is not None else r['spectrum_raw_y']
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        if len(x) < 10:
            x = np.asarray(r['spectrum_raw_x'], dtype=float); y = np.asarray(r['spectrum_raw_y'], dtype=float)
        yg = resample(x, y)
        rec = {'smiles': smi, 'kekule_smiles': kek, 'formula': str(r['formula']),
               'name': str(r['name']), 'cas': str(r['cas']),
               'ir_grid': yg, 'xmin': float(np.min(x)), 'xmax': float(np.max(x))}
        for k, v in norm_variants(yg).items():
            rec['ir_' + k] = v
        recs.append(rec)
    print('records', len(recs), 'skipped', skipped)
    with open('datasets/external_irspec/external_ir.pkl', 'wb') as f:
        pickle.dump(recs, f)
    yg = np.array([r['ir_grid'] for r in recs])
    print('grid shape', yg.shape, 'max mean', float(yg.max(1).mean()), 'sum mean', float(yg.sum(1).mean()))
    print('unique kekule smiles', len(set(r['kekule_smiles'] for r in recs)))
