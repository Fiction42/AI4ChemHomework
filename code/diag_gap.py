# -*- coding: utf-8 -*-
"""验证外部数据 domain gap: 对重叠分子比较 (a) ViBench 谱嵌入与自身分子嵌入的相似度,
(b) 外部谱嵌入与同一分子嵌入的相似度, 并给出两类谱的形状统计。"""
import os, sys, pickle
import numpy as np, torch, lmdb
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
REPO='/Users/rongronghehe/Desktop/工作:学习/Exam/ref/vib2mol'
sys.path.insert(0, REPO); os.chdir(REPO)
from models import build_model
from utils.collators import BaseCollator
GRID=np.linspace(400,4000,1024)

def ik(s):
    m=Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m) if m else None
def norm(y):
    y=np.clip(np.asarray(y,float),0,None); m=y.max(); return y/m if m>0 else y
class DS(torch.utils.data.Dataset):
    def __init__(self,specs,smis): self.specs,self.smis=specs,smis
    def __len__(self): return len(self.smis)
    def __getitem__(self,i): return {'ir':torch.as_tensor(self.specs[i],dtype=torch.float32),'smiles':self.smis[i]}
def collate(b): return BaseCollator(tokenizer_path='models/MolTokenizer',spectral_types=['ir'])(b)
def embed(model,specs,smis):
    loader=torch.utils.data.DataLoader(DS(specs,smis),batch_size=16,collate_fn=collate,shuffle=False)
    sp,mo=[],[]; model.eval()
    with torch.no_grad():
        for b in loader:
            d=b['data']; sp.append(model.get_spectral_embeddings(d)['proj_output']); mo.append(model.get_molecular_embeddings(d,use_cls_token=True)['proj_output'])
    return torch.cat(sp), torch.cat(mo)

ck='/Users/rongronghehe/Desktop/工作:学习/Exam/checkpoints/sdbs/ir-kekule_smiles/vib2mol/2025-07-16-15-32-ea6ef7/epoch999.pth'
model=build_model('vib2mol',spectral_channel=1)
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(ck,map_location='cpu',weights_only=True).items()},strict=True); model.eval()

# ViBench test
vs, vmm = [], []
env=lmdb.open('/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/vibench/sdbs/sdbs_test.lmdb',subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
with env.begin() as txn:
    for k,v in txn.cursor().iternext():
        d=pickle.loads(v); vs.append(norm(d['ir'])); vmm.append(d['kekule_smiles'])
env.close()
sp_v, mo_v = embed(model, vs, vmm)
Simv = (torch.nn.functional.normalize(sp_v,dim=1) @ torch.nn.functional.normalize(mo_v,dim=1).t()).numpy()
diag_v = np.diag(Simv)
print('ViBench: diag sim mean %.3f  min %.3f  max %.3f' % (diag_v.mean(), diag_v.min(), diag_v.max()))
print('ViBench: frac of points >0.1: %.3f' % np.mean([(s>0.1).mean() for s in vs]))

# external overlap
ext=pickle.load(open('/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/external_irspec/external_ir.pkl','rb'))
viks={ik(s) for s in vmm}
sel=[r for r in ext if ik(r['kekule_smiles']) in viks]
seen=set(); sel2=[]
for r in sel:
    k=ik(r['kekule_smiles'])
    if k in seen: continue
    seen.add(k); sel2.append(r)
print('overlap unique', len(sel2))
es=[norm(r['ir_grid']) for r in sel2]; esm=[r['kekule_smiles'] for r in sel2]
sp_e, mo_e = embed(model, es, esm)
print('external: frac of points >0.1: %.3f' % np.mean([(s>0.1).mean() for s in es]))
# 对每个重叠分子: ViBench 谱 vs 外部谱 嵌入相似度
name2idx={}
for i,s in enumerate(vmm):
    name2idx.setdefault(ik(s), i)
d_spec=[]
for j,r in enumerate(sel2):
    i=name2idx[ik(r['kekule_smiles'])]
    a=torch.nn.functional.normalize(sp_v[i:i+1],dim=1); b=torch.nn.functional.normalize(sp_e[j:j+1],dim=1)
    d_spec.append(float(a@b.t()))
print('same molecule ViBench-IR vs external-IR embedding cos: mean %.3f' % np.mean(d_spec))
# 各自与正确分子嵌入的相似度
vals_e=[]
for j,r in enumerate(sel2):
    i=name2idx[ik(r['kekule_smiles'])]
    a=torch.nn.functional.normalize(sp_e[j:j+1],dim=1); m=torch.nn.functional.normalize(mo_v[i:i+1],dim=1)
    vals_e.append(float(a@m.t()))
print('external spectrum -> its molecule (via ViBench mol emb): mean %.3f' % np.mean(vals_e))
# 同一分子 ViBench 谱 -> 分子
vals_v=[float(np.dot(torch.nn.functional.normalize(sp_v[i:i+1],dim=1).numpy()[0], torch.nn.functional.normalize(mo_v[i:i+1],dim=1).numpy()[0])) for i in [name2idx[ik(r['kekule_smiles'])] for r in sel2]]
print('ViBench spectrum -> molecule: mean %.3f' % np.mean(vals_v))
# 谱形状对比: 局部峰数(局部极大)与噪声
def stats(s):
    loc = np.sum((s[1:-1]>s[:-2]) & (s[1:-1]>s[2:]) & (s[1:-1]>0.05))
    return loc, float(np.std(s)), float(np.mean(s))
print('ViBench local maxima mean %.1f | external mean %.1f' % (np.mean([stats(s)[0] for s in vs]), np.mean([stats(s)[0] for s in es])))
