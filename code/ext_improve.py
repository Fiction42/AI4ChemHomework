# -*- coding: utf-8 -*-
"""改进实验: (1) 全局化学式过滤; (2) 少样本域自适应 (只训练投影头);
(3) 与基线/hard-case 对比; 输出 JSON 与绘图数据。"""
import os, sys, json, pickle
import numpy as np, torch, torch.nn as nn
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
REPO='/Users/rongronghehe/Desktop/工作:学习/Exam/ref/vib2mol'
sys.path.insert(0, REPO); os.chdir(REPO)
from models import build_model
from utils.collators import BaseCollator
GRID=np.linspace(400,4000,1024); OUT='/Users/rongronghehe/Desktop/工作:学习/Exam/results'

def ik(s):
    m=Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m) if m else None
def norm(y):
    y=np.clip(np.asarray(y,float),0,None); m=y.max(); return y/m if m>0 else y
class DS(torch.utils.data.Dataset):
    def __init__(self,specs,smis): self.specs,self.smis=specs,smis
    def __len__(self): return len(self.smis)
    def __getitem__(self,i): return {'ir':torch.as_tensor(self.specs[i],dtype=torch.float32),'smiles':self.smis[i]}
def collate(b): return BaseCollator(tokenizer_path='models/MolTokenizer',spectral_types=['ir'])(b)

def embeds(model, specs, smis, cls768=False):
    loader=torch.utils.data.DataLoader(DS(specs,smis),batch_size=16,collate_fn=collate,shuffle=False)
    sp,mo=[],[]
    model.eval()
    with torch.no_grad():
        for b in loader:
            d=b['data']
            sp.append(model.get_spectral_embeddings(d)['proj_output'])
            mo.append(model.get_molecular_embeddings(d,use_cls_token=True)['proj_output'])
    return torch.cat(sp).numpy(), torch.cat(mo).numpy()

def recall(S, k):
    n=S.shape[0]; top=np.argsort(-S,axis=1)[:,:k]; hit=sum(int(i in top[i]) for i in range(n)); return hit/n

def formula_recall(S, forms, k):
    n=S.shape[0]; idx={}
    for i,f in enumerate(forms): idx.setdefault(f,[]).append(i)
    hits={1:0,3:0,5:0}; sizes=[]
    for i in range(n):
        grp=idx[forms[i]]; sizes.append(len(grp))
        order=sorted(grp, key=lambda j: -S[i,j])
        for kk in (1,3,5):
            if i in order[:kk]: hits[kk]+=1
    return {'Recall@1':hits[1]/n,'Recall@3':hits[3]/n,'Recall@5':hits[5]/n,
            'mean_group_size': float(np.mean(sizes))}

def main():
    ext=pickle.load(open('/Users/rongronghehe/Desktop/工作:学习/Exam/datasets/external_irspec/external_ir.pkl','rb'))
    ext=[r for r in ext if ik(r['kekule_smiles'])]
    seen=set(); recs=[]
    for r in ext:
        k=ik(r['kekule_smiles'])
        if k in seen: continue
        seen.add(k); recs.append(r)
    rng=np.random.default_rng(0); perm=rng.permutation(len(recs))
    n_adapt=600
    adapt=[recs[i] for i in perm[:n_adapt]]; ev=[recs[i] for i in perm[n_adapt:]]
    print('adapt',len(adapt),'eval',len(ev), flush=True)
    ck='/Users/rongronghehe/Desktop/工作:学习/Exam/checkpoints/sdbs/ir-kekule_smiles/vib2mol/2025-07-16-15-32-ea6ef7/epoch999.pth'
    model=build_model('vib2mol',spectral_channel=1)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(ck,map_location='cpu',weights_only=True).items()},strict=True); model.eval()
    res={}
    # 基线
    sp,mo=embeds(model,[norm(r['ir_grid']) for r in recs],[r['kekule_smiles'] for r in recs])
    forms=[r['formula'] for r in recs]
    spn=sp/np.linalg.norm(sp,axis=1,keepdims=True)
    mon=mo/np.linalg.norm(mo,axis=1,keepdims=True)
    S=spn@mon.T
    ev_idx=np.asarray(perm[n_adapt:])
    S2=S[np.ix_(ev_idx, ev_idx)]
    ev_forms=[forms[i] for i in ev_idx]
    res['baseline_eval']={'Recall@1':recall(S2,1),'Recall@3':recall(S2,3),'Recall@5':recall(S2,5)}
    res['baseline_eval_formula']=formula_recall(S2, ev_forms, 5)
    np.save(f'{OUT}/imp_S2_baseline.npy', S2)
    # 少样本投影头适配
    orig_proj=model.spectral_encoder.proj
    model.spectral_encoder.proj=nn.Identity()
    sp_adapt768,_=embeds(model,[norm(r['ir_grid']) for r in adapt],[r['kekule_smiles'] for r in adapt])
    sp_ev768,_=embeds(model,[norm(r['ir_grid']) for r in ev],[r['kekule_smiles'] for r in ev])
    _, mo_ev=embeds(model,[norm(r['ir_grid']) for r in ev],[r['kekule_smiles'] for r in ev])
    model.spectral_encoder.proj=orig_proj
    Xa=torch.tensor(sp_adapt768,dtype=torch.float32)
    Xe=torch.tensor(sp_ev768,dtype=torch.float32)
    proj=nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Linear(256,256))
    opt=torch.optim.Adam(proj.parameters(),lr=1e-3)
    Mo=torch.tensor(mo_ev,dtype=torch.float32)
    # 负样本: 用 adapt 分子嵌入
    _, mo_adapt=embeds(model,[norm(r['ir_grid']) for r in adapt],[r['kekule_smiles'] for r in adapt])
    Madapt=torch.tensor(mo_adapt,dtype=torch.float32)
    import torch.nn.functional as F
    for ep in range(60):
        opt.zero_grad()
        z=F.normalize(proj(Xa),dim=1); m=F.normalize(Madapt,dim=1)
        logits=z@m.t()/0.07
        loss=F.cross_entropy(logits, torch.arange(len(Xa)))
        loss.backward(); opt.step()
    with torch.no_grad():
        ze=F.normalize(proj(Xe),dim=1).numpy()
    # 分子嵌入用分子编码器输出 (对 eval 分子)
    mon_ev=mo_ev/np.linalg.norm(mo_ev,axis=1,keepdims=True)
    S3=ze@mon_ev.T
    res['fewshot_proj_eval']={'Recall@1':recall(S3,1),'Recall@3':recall(S3,3),'Recall@5':recall(S3,5)}
    res['fewshot_proj_eval_formula']=formula_recall(S3, ev_forms, 5)
    np.save(f'{OUT}/imp_S3_fewshot.npy', S3)
    res['n_adapt']=n_adapt; res['n_eval']=len(ev_idx)
    json.dump(res,open(f'{OUT}/ext_improvements2.json','w'),indent=1,ensure_ascii=False)
    print(json.dumps(res,indent=1,ensure_ascii=False))

if __name__=='__main__':
    main()