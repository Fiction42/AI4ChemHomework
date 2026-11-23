# -*- coding: utf-8 -*-
"""文献外独立数据集 (Lamblador/IRSpectra) 的分析: 公式过滤、TTA 改进与成功/失败案例。"""
import os, sys, json, pickle, collections
import numpy as np, torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
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

def formula(s):
    m = Chem.MolFromSmiles(s)
    return rdMolDescriptors.CalcMolFormula(m) if m else None

def norm(y):
    y = np.clip(np.asarray(y, float), 0, None); m = y.max()
    return y / m if m > 0 else y

def shift(y, d):
    return np.interp(GRID, GRID + d, y, left=0.0, right=0.0)

def smooth(y, sig):
    if sig <= 0: return y
    k = np.exp(-0.5 * (np.arange(-3*sig, 3*sig+1) / sig**2 * sig) ** 2)
    k = np.exp(-0.5 * (np.arange(-3*sig, 3*sig+1) / sig) ** 2); k /= k.sum()
    return np.convolve(y, k, mode='same')

def tanimoto(a, b):
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None: return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))

class DS(torch.utils.data.Dataset):
    def __init__(self, specs, smis): self.specs, self.smis = specs, smis
    def __len__(self): return len(self.smis)
    def __getitem__(self, i):
        return {'ir': torch.as_tensor(self.specs[i], dtype=torch.float32), 'smiles': self.smis[i]}

def collate(b): return BaseCollator(tokenizer_path='models/MolTokenizer', spectral_types=['ir'])(b)

def embed(model, specs, smis, bs=16):
    loader = torch.utils.data.DataLoader(DS(specs, smis), batch_size=bs, collate_fn=collate, shuffle=False)
    sp, mo = [], []
    model.eval()
    with torch.no_grad():
        for b in loader:
            d = b['data']
            sp.append(model.get_spectral_embeddings(d)['proj_output'])
            mo.append(model.get_molecular_embeddings(d, use_cls_token=True)['proj_output'])
    return torch.cat(sp), torch.cat(mo)

def recall_at(S, k, formulas=None, filter_formula=False, topk_pool=50):
    n = S.shape[0]
    order = np.argsort(-S, axis=1)
    if filter_formula:
        order = order[:, :topk_pool]
    out = {}
    for kk in (1, 3, 5, 10):
        hit = 0
        for i in range(n):
            cand = order[i, :min(kk, order.shape[1])]
            if filter_formula:
                cand = [j for j in cand if formulas[j] == formulas[i]]
                cand = cand[:kk] if cand else []
            ok = (i in list(cand))
            hit += 1 if ok else 0
        out[f'Recall@{kk}'] = hit / n
    return out

def main():
    ext = pickle.load(open('/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/external_irspec/external_ir.pkl', 'rb'))
    ext = [r for r in ext if ik(r['kekule_smiles'])]
    seen, recs = set(), []
    for r in ext:
        k = ik(r['kekule_smiles'])
        if k in seen: continue
        seen.add(k); recs.append(r)
    specs = [norm(r['ir_grid']) for r in recs]
    smis = [r['kekule_smiles'] for r in recs]
    forms = [r['formula'] for r in recs]
    print('n', len(recs), flush=True)

    ck = '/Users/rongronghehe/Desktop/工作:学习/Exam/checkpoints/sdbs/ir-kekule_smiles/vib2mol/2025-07-16-15-32-ea6ef7/epoch999.pth'
    model = build_model('vib2mol', spectral_channel=1)
    model.load_state_dict({k.replace('module.',''): v for k, v in torch.load(ck, map_location='cpu', weights_only=True).items()}, strict=True)
    model.eval()

    res = {}
    sp, mo = embed(model, specs, smis)
    S = (torch.nn.functional.normalize(sp, dim=1) @ torch.nn.functional.normalize(mo, dim=1).t()).numpy()
    res['baseline'] = recall_at(S, 1)
    res['formula_filter'] = recall_at(S, 1, formulas=forms, filter_formula=True)
    res['formula_filter_pool20'] = recall_at(S, 1, formulas=forms, filter_formula=True, topk_pool=20)
    # TTA: 平移与平滑平均
    tta_list = []
    for d in (-10, -5, 0, 5, 10):
        spd, _ = embed(model, [shift(s, d) for s in specs], smis)
        tta_list.append(torch.nn.functional.normalize(spd, dim=1).numpy())
    sp_tta = np.mean(tta_list, axis=0)
    S_tta = sp_tta @ torch.nn.functional.normalize(mo, dim=1).numpy().T
    res['tta_shift'] = recall_at(S_tta, 1)
    res['tta_shift_formula'] = recall_at(S_tta, 1, formulas=forms, filter_formula=True)
    np.save(f'{OUT}/ext_spectral_emb.npy', sp.numpy()); np.save(f'{OUT}/ext_mol_emb.npy', mo.numpy())
    json.dump(res, open(f'{OUT}/ext_improvements.json', 'w'), indent=1, ensure_ascii=False)

    # 案例: 每条的 top-1 与 top-5
    order = np.argsort(-S, axis=1)
    cases = []
    for i in range(len(recs)):
        top = order[i, :5].tolist()
        cases.append({
            'i': i, 'name': recs[i].get('name',''), 'cas': recs[i].get('cas',''),
            'formula': forms[i], 'true_smiles': smis[i],
            'top1_smiles': smis[top[0]], 'top1_sim': float(S[i, top[0]]),
            'top5_smiles': [smis[j] for j in top], 'top5_sim': [float(S[i, j]) for j in top],
            'top1_correct': bool(top[0] == i),
            'top5_correct': bool(i in top),
            'tanimoto_top1': tanimoto(smis[i], smis[top[0]]),
            'formula_top1_match': bool(forms[top[0]] == forms[i]),
            'n_peaks': int(np.sum(specs[i] > 0.1)),
            'xmin': recs[i]['xmin'], 'xmax': recs[i]['xmax'],
        })
    json.dump(cases, open(f'{OUT}/ext_cases_analysis.json', 'w'), ensure_ascii=False, indent=1)
    # 统计: 正确/错误案例的谱特征
    corr = [c for c in cases if c['top1_correct']]
    wrong = [c for c in cases if not c['top1_correct']]
    stats = {
        'n_correct_top1': len(corr), 'n_wrong_top1': len(wrong),
        'mean_npeaks_correct': float(np.mean([c['n_peaks'] for c in corr])) if corr else None,
        'mean_npeaks_wrong': float(np.mean([c['n_peaks'] for c in wrong])),
        'mean_tanimoto_wrong': float(np.mean([c['tanimoto_top1'] for c in wrong if c['tanimoto_top1'] is not None])),
        'frac_formula_match_wrong': float(np.mean([c['formula_top1_match'] for c in wrong])),
        'mean_sim_correct': float(np.mean([c['top1_sim'] for c in corr])) if corr else None,
        'mean_sim_wrong': float(np.mean([c['top1_sim'] for c in wrong])),
        'fraction_xmax_lt_3700': float(np.mean([c['xmax'] < 3700 for c in cases])),
    }
    json.dump(stats, open(f'{OUT}/ext_case_stats.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1))
    print(json.dumps(stats, indent=1))

if __name__ == '__main__':
    main()
