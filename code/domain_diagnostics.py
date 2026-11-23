# -*- coding: utf-8 -*-
"""诊断: (A) 用同一脚本复现 ViBench SDBS test 的检索, 验证管线正确性;
(B) 文献外数据在不同波数平移/平滑下的检索表现; (C) 在重叠分子子集上的检索。"""
import os, sys, json, pickle
import numpy as np, torch, lmdb
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

REPO = '/Users/rongronghehe/Desktop/工作:学习/Exam/ref/vib2mol'
sys.path.insert(0, REPO); os.chdir(REPO)
from models import build_model
from utils.collators import BaseCollator

GRID = np.linspace(400, 4000, 1024)
OUT = '/Users/rongronghehe/Desktop/工作:学习/Exam/results'

def ik(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m) if m else None

def norm(y):
    y = np.clip(np.asarray(y, float), 0, None)
    m = y.max()
    return y / m if m > 0 else y

def shift_spec(y, dcm):
    """波数整体平移 dcm cm^-1 (等价于在原始波数轴上平移后重采样)"""
    x = GRID + dcm
    return np.interp(GRID, x, y, left=0.0, right=0.0)

def smooth(y, sigma_pts):
    if sigma_pts <= 0:
        return y
    k = np.exp(-0.5 * (np.arange(-3*sigma_pts, 3*sigma_pts+1) / sigma_pts) ** 2)
    k = k / k.sum()
    return np.convolve(y, k, mode='same')

class DS(torch.utils.data.Dataset):
    def __init__(self, specs, smis):
        self.specs, self.smis = specs, smis
    def __len__(self): return len(self.smis)
    def __getitem__(self, i):
        return {'ir': torch.as_tensor(self.specs[i], dtype=torch.float32), 'smiles': self.smis[i]}

def collate(batch):
    return BaseCollator(tokenizer_path='models/MolTokenizer', spectral_types=['ir'])(batch)

def embed_all(model, specs, smis, bs=16):
    loader = torch.utils.data.DataLoader(DS(specs, smis), batch_size=bs, collate_fn=collate, shuffle=False)
    sp, mo = [], []
    model.eval()
    with torch.no_grad():
        for b in loader:
            d = b['data']
            s = model.get_spectral_embeddings(d)['proj_output']
            m = model.get_molecular_embeddings(d, use_cls_token=True)['proj_output']
            sp.append(s); mo.append(m)
    return torch.cat(sp), torch.cat(mo)

def recall(sp, mo, inchis, cand_idx=None):
    S = torch.nn.functional.normalize(sp, dim=1) @ torch.nn.functional.normalize(mo, dim=1).t()
    S = S.numpy()
    if cand_idx is not None:
        S = S[:, np.asarray(cand_idx)]
        target = {c: j for j, c in enumerate(cand_idx)}
    else:
        target = {i: i for i in range(S.shape[0])}
    top = np.argsort(-S, axis=1)
    res = {}
    for k in (1, 3, 5, 10):
        hit = 0; tot = 0
        for i in range(S.shape[0]):
            if cand_idx is not None and i not in target:
                continue
            tot += 1
            want = target.get(i, None)
            oks = [inchis[cand_idx[j]] if cand_idx is not None else inchis[j] for j in top[i, :k]]
            if i in top[i, :k] and (cand_idx is None or want is not None):
                hit += 1
            elif inchis[i] in oks:
                hit += 1
        res[f'Recall@{k}'] = hit / max(tot, 1)
    res['n_query'] = tot
    return res

def main():
    ckpt = '/Users/rongronghehe/Desktop/工作:学习/Exam/checkpoints/sdbs/ir-kekule_smiles/vib2mol/2025-07-16-15-32-ea6ef7/epoch999.pth'
    model = build_model('vib2mol', spectral_channel=1)
    ck = {k.replace('module.', ''): v for k, v in torch.load(ckpt, map_location='cpu', weights_only=True).items()}
    model.load_state_dict(ck, strict=True); model.eval()
    out = {}
    # A: in-domain
    specs, smis = [], []
    env = lmdb.open('/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/vibench/sdbs/sdbs_test.lmdb', subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        for k, v in txn.cursor().iternext():
            d = pickle.loads(v)
            specs.append(norm(d['ir'])); smis.append(d['kekule_smiles'])
    env.close()
    inch = [ik(s) for s in smis]
    sp, mo = embed_all(model, specs, smis)
    out['A_indomain'] = recall(sp, mo, inch)
    print('A in-domain', out['A_indomain'], flush=True)
    # B: external with shift/smooth
    ext = pickle.load(open('/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/external_irspec/external_ir.pkl', 'rb'))
    ext = [r for r in ext if ik(r['kekule_smiles']) is not None]
    # 去重(按 InChIKey 保留第一条)
    seen = set(); ext_u = []
    for r in ext:
        kk = ik(r['kekule_smiles'])
        if kk in seen: continue
        seen.add(kk); ext_u.append(r)
    base_specs = [norm(r['ir_grid']) for r in ext_u]
    base_smis = [r['kekule_smiles'] for r in ext_u]
    inch_ext = [ik(s) for s in base_smis]
    out['B_external'] = {}
    for dcm in (0, -20, -10, 10, 20, 40):
        specs = [shift_spec(norm(s), dcm) for s in base_specs]
        sp, mo = embed_all(model, specs, base_smis)
        r = recall(sp, mo, inch_ext)
        out['B_external'][f'shift_{dcm}'] = r
        print('B shift', dcm, r, flush=True)
    for sig in (0, 1, 2, 4):
        specs = [smooth(norm(s), sig) for s in base_specs]
        sp, mo = embed_all(model, specs, base_smis)
        out['B_external'][f'smooth_{sig}'] = recall(sp, mo, inch_ext)
        print('B smooth', sig, out['B_external'][f'smooth_{sig}'], flush=True)
    # C: overlap subset
    vib_iks = set(inch)
    idx = [i for i, k in enumerate(inch_ext) if k in vib_iks]
    if len(idx) > 5:
        sp, mo = embed_all(model, base_specs, base_smis)
        out['C_overlap'] = recall(sp, mo, inch_ext, cand_idx=idx)
        print('C overlap', out['C_overlap'], 'n', len(idx), flush=True)
    json.dump(out, open(f'{OUT}/domain_diagnostics.json', 'w'), indent=1, ensure_ascii=False)

if __name__ == '__main__':
    main()
