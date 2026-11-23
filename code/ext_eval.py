# -*- coding: utf-8 -*-
"""文献外独立 IR 数据集上的 Vib2Mol 迁移测试。
- 数据集: HuggingFace Lamblador/IRSpectra (test, 2128 条实验 IR 光谱, 与 ViBench 无重叠来源)
- 任务: 闭集检索 (IR 谱 -> 分子), 与 ViBench SDBS test 的同协议结果对比
- 输出: results/ext_eval_<tag>.json 与 results/ext_cases_<tag>.json
"""
import os, sys, json, pickle, argparse
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

REPO = os.path.abspath('ref/vib2mol')
sys.path.insert(0, REPO)
os.chdir(REPO)

from models import build_model
from utils.collators import BaseCollator

GRID = np.linspace(400, 4000, 1024)

def norm_spectrum(y, mode):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, None)
    if mode == 'max1':
        m = y.max()
        return y / m if m > 0 else y
    if mode == 'sum1':
        s = y.sum()
        return y / s * 100.0 if s > 0 else y
    if mode == 'l2_1':
        n = np.linalg.norm(y)
        return y / n * 10.0 if n > 0 else y
    return y

def inchi_key(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return None

def tanimoto(s1, s2):
    m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))

class RecDataset(torch.utils.data.Dataset):
    def __init__(self, recs, key='ir'):
        self.recs = recs
        self.key = key
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return {'ir': torch.as_tensor(r[self.key], dtype=torch.float32),
                'smiles': r['kekule_smiles']}

def collate(batch):
    collator = BaseCollator(tokenizer_path='models/MolTokenizer', spectral_types=['ir'])
    return collator(batch)

def embed(model, loader, device='cpu', ir_only=False):
    model.eval()
    sp, mo = [], []
    with torch.no_grad():
        for batch in loader:
            data = batch['data']
            data = {k: {kk: (vv.to(device) if hasattr(vv, 'to') else vv) for kk, vv in v.items()} if isinstance(v, dict) else v
                    for k, v in data.items()}
            s = model.get_spectral_embeddings(data, raman_only=False, ir_only=ir_only)['proj_output']
            m = model.get_molecular_embeddings(data, use_cls_token=True)['proj_output']
            sp.append(s.cpu()); mo.append(m.cpu())
    return torch.cat(sp), torch.cat(mo)

def recall_from_topk(topk_idx, n, n_inchi, k_list=(1,3,5,10), inchis=None):
    out = {}
    for k in k_list:
        if k > topk_idx.shape[1]:
            continue
        idx_hit = sum(1 for i in range(n) if i in topk_idx[i, :k].tolist())
        out[f'Recall@{k}'] = idx_hit / n
        if inchis is not None:
            hit = 0
            for i in range(n):
                if inchis[i] in [inchis[j] for j in topk_idx[i, :k].tolist()]:
                    hit += 1
            out[f'MolRecall@{k}'] = hit / n
    return out

def load_records(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/external_irspec/external_ir.pkl')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arch', default='vib2mol')
    ap.add_argument('--tag', default='vib2mol')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--ir-only', action='store_true')
    args = ap.parse_args()

    recs = load_records(args.records)
    outdir = '/Users/rongronghehe/Desktop/工作:学习/Exam/results'
    os.makedirs(outdir, exist_ok=True)
    results = {'n_records': len(recs), 'ckpt': args.ckpt, 'arch': args.arch}
    inchis = [inchi_key(r['kekule_smiles']) for r in recs]
    results['n_unique_inchikey'] = len({x for x in inchis if x})

    spectral_channel = 2 if args.arch.endswith('mmm') else 1
    model = build_model(args.arch, spectral_channel=spectral_channel)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    for mode in ['max1', 'sum1', 'l2_1', 'raw']:
        for r in recs:
            r['ir'] = norm_spectrum(r['ir_grid'], mode)
        ds = RecDataset(recs)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, shuffle=False)
        sp, mo = embed(model, loader, ir_only=args.ir_only)
        sim = torch.nn.functional.normalize(sp, dim=1) @ torch.nn.functional.normalize(mo, dim=1).t()
        sim_np = sim.numpy()
        kk = min(args.topk, sim_np.shape[1])
        topk_idx = np.argsort(-sim_np, axis=1)[:, :kk]
        m = recall_from_topk(topk_idx, len(recs), len(recs), k_list=(1,3,5,10), inchis=inchis)
        m['mean_top1_sim'] = float(sim_np[np.arange(len(recs)), topk_idx[:, 0]].mean())
        results[mode] = m
        if mode == 'max1':
            cases = []
            for i in range(len(recs)):
                pred = topk_idx[i, :5].tolist()
                cases.append({
                    'i': i, 'name': recs[i]['name'], 'cas': recs[i]['cas'],
                    'formula': recs[i]['formula'], 'true_smiles': recs[i]['kekule_smiles'],
                    'pred_smiles': recs[i]['kekule_smiles'] if False else recs[pred[0]]['kekule_smiles'],
                    'top5_idx': pred,
                    'top5_smiles': [recs[j]['kekule_smiles'] for j in pred],
                    'top1_sim': float(sim_np[i, pred[0]]),
                    'top5_sim': [float(sim_np[i, j]) for j in pred],
                    'correct_top1': bool(pred[0] == i or (inchis[pred[0]] is not None and inchis[pred[0]] == inchis[i])),
                    'tanimoto_top1': tanimoto(recs[i]['kekule_smiles'], recs[pred[0]]['kekule_smiles']),
                })
            with open(f'{outdir}/ext_cases_{args.tag}.json', 'w') as f:
                json.dump(cases, f, ensure_ascii=False, indent=1)
        print('[%s] %s done' % (args.tag, mode), flush=True)
    with open(f'{outdir}/ext_eval_{args.tag}.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in results.items() if isinstance(v, dict)}, indent=1)[:2000])

if __name__ == '__main__':
    main()