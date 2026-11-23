# -*- coding: utf-8 -*-
import os, sys, pickle
import numpy as np, lmdb
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

def ik(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m) if m else None

ext = pickle.load(open('/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/external_irspec/external_ir.pkl','rb'))
ext_by_ik = {}
for r in ext:
    k = ik(r['kekule_smiles'])
    if k and k not in ext_by_ik:
        ext_by_ik[k] = r
print('external unique', len(ext_by_ik))

vib = {}
for split in ['test','train']:
    env = lmdb.open(f'/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/vibench/sdbs/sdbs_{split}.lmdb', subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        c = txn.cursor()
        for k, v in c.iternext():
            d = pickle.loads(v)
            kk = ik(d['kekule_smiles'])
            if kk and kk not in vib:
                vib[kk] = {'split': split, 'ir': np.asarray(d['ir']), 'raman': np.asarray(d.get('raman')) if d.get('raman') is not None else None, 'smiles': d['kekule_smiles']}
    env.close()
print('vibench sdbs unique', len(vib))
common = set(ext_by_ik) & set(vib)
print('overlap molecules', len(common))
ids = list(common)[:3]
grid = np.linspace(400,4000,1024)
for k in ids:
    e = ext_by_ik[k]; v = vib[k]
    ey = np.interp(grid, np.asarray(e['ir_grid']), np.zeros_like(grid)) if False else np.asarray(e['ir_grid'])
    ey = ey / (ey.max() if ey.max()>0 else 1)
    vy = np.asarray(v['ir']); vy = vy/vy.max()
    c1 = float(np.corrcoef(ey, vy)[0,1])
    c2 = float(np.corrcoef(1-ey, vy)[0,1])
    c3 = float(np.corrcoef(-ey, vy)[0,1])
    print('---', k, e['name'], '| split', v['split'])
    print('  ext max/sum', round(float(ey.max()),3), round(float(ey.sum()),2), 'vib max/sum', round(float(vy.max()),3), round(float(vy.sum()),2))
    print('  corr(ext,vib)=%.3f  corr(1-ext,vib)=%.3f  corr(-ext,vib)=%.3f' % (c1, c2, c3))
    # peak positions
    print('  ext top peaks', grid[np.argsort(ey)[-5:]].round(0), ' vib top peaks', grid[np.argsort(vy)[-5:]].round(0))
    print('  ext baseline(median)', round(float(np.median(ey)),3), ' vib baseline(median)', round(float(np.median(vy)),3))
