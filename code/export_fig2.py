# -*- coding: utf-8 -*-
"""导出作业2 报告所需作图数据 (CSV) 与案例信息。"""
import os, json, pickle
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT='/Users/rongronghehe/Desktop/工作:学习/Exam'
OUT=f'{ROOT}/figs2'; os.makedirs(OUT, exist_ok=True)
GRID=np.linspace(400,4000,1024)

ext=pickle.load(open(f'{ROOT}/datasets/external_irspec/external_ir.pkl','rb'))
def norm(y):
    y=np.clip(np.asarray(y,float),0,None); m=y.max(); return y/m if m>0 else y
recs=[]; seen=set()
for r in ext:
    m=Chem.MolFromSmiles(r['kekule_smiles'])
    if m is None: continue
    k=Chem.MolToInchiKey(m)
    if k in seen: continue
    seen.add(k); recs.append(r)
print('records', len(recs))

cases=json.load(open(f'{ROOT}/results/ext_cases_analysis.json'))
def find(namepart):
    for c in cases:
        if namepart.lower() in c['name'].lower():
            return c
    return None
sel=[('succ1','CH3SCH2CN'),('succ2',"1,1'-Biphenyl, 2-chloro-"),
     ('fail1','diundecyl'),('fail2','Benzaldehyde, 4-methyl-')]
meta={}
with open(f'{OUT}/case_spectra.csv','w') as f:
    f.write('case,wave,y\n')
    for tag,pat in sel:
        c=find(pat)
        if c is None:
            print('missing', tag, pat); continue
        y=norm(recs[c['i']]['ir_grid'])
        for x,v in zip(GRID,y):
            f.write(f'{tag},{x:.2f},{v:.5f}\n')
        meta[tag]={'name':c['name'],'formula':c['formula'],'true':c['true_smiles'],
                   'pred':c['top1_smiles'],'sim':c['top1_sim'],'tanimoto':c['tanimoto_top1'],
                   'correct':c['top1_correct'],'cas':c['cas']}
json.dump(meta, open(f'{OUT}/case_meta.json','w'), ensure_ascii=False, indent=1)

# 同一分子在 ViBench 与外部数据集的谱对比
import lmdb
target=find('diundecyl')
# 选一个在 ViBench test 中存在的化合物做对比
vib={}
env=lmdb.open(f'{ROOT}/datasets/vibench/sdbs/sdbs_test.lmdb',subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
with env.begin() as txn:
    for k,v in txn.cursor().iternext():
        d=pickle.loads(v)
        m=Chem.MolFromSmiles(d['kekule_smiles'])
        vib[Chem.MolToInchiKey(m)]=(d['kekule_smiles'],np.asarray(d['ir']))
env.close()
pair=None
for r in recs:
    m=Chem.MolFromSmiles(r['kekule_smiles'])
    k=Chem.MolToInchiKey(m)
    if k in vib:
        pair=(r, vib[k]); break
r,vv=pair
with open(f'{OUT}/overlap_spectra.csv','w') as f:
    f.write('wave,vibench,external\n')
    ye=norm(r['ir_grid']); yv=norm(vv[1])
    for x,a,b in zip(GRID,yv,ye):
        f.write(f'{x:.2f},{a:.5f},{b:.5f}\n')
json.dump({'smiles':vv[0],'name':r.get('name','')}, open(f'{OUT}/overlap_meta.json','w'), ensure_ascii=False, indent=1)

# 平移/平滑敏感性
dd=json.load(open(f'{ROOT}/results/domain_diagnostics.json'))
with open(f'{OUT}/shift_sens.csv','w') as f:
    f.write('shift,Recall@1,Recall@5\n')
    for k,v in dd['B_external'].items():
        if k.startswith('shift_'):
            f.write(f"{k.split('_')[1]},{v['Recall@1']:.5f},{v['Recall@5']:.5f}\n")
with open(f'{OUT}/smooth_sens.csv','w') as f:
    f.write('sigma,Recall@1,Recall@5\n')
    for k,v in dd['B_external'].items():
        if k.startswith('smooth_'):
            f.write(f"{k.split('_')[1]},{v['Recall@1']:.5f},{v['Recall@5']:.5f}\n")

# Tanimoto 分布 (错误案例)
w=[c['tanimoto_top1'] for c in cases if not c['top1_correct'] and c['tanimoto_top1'] is not None]
cnt,edges=np.histogram(w,bins=20,range=(0,1))
with open(f'{OUT}/tanimoto_hist.csv','w') as f:
    f.write('lo,hi,count\n')
    for i in range(len(cnt)):
        f.write(f'{edges[i]:.3f},{edges[i+1]:.3f},{cnt[i]}\n')
print('exported', os.listdir(OUT))
