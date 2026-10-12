"""Independent re-derivation of the per-origin DM test, with Bartlett (p), rectangular
(p2) and flattened (p3) variance kernels. Run from anywhere; writes dm_kernel_check.csv here.
The rectangular column is the source of the kernel-sensitivity counts in the paper."""
import numpy as np, glob
from pathlib import Path
from scipy import stats
HERE=Path(__file__).resolve().parent
P=str(HERE.parents[3]/"outputs"/"predictions")+"/"
DS=["cesnet","abilene","geant","nab_aws_cpu","nab_twitter"]; HS=[1,3,6,12,24]
BL=["chronos_bolt_zs","chronos_zs","moirai_zs","timesfm_zs","ttm_zs","patchtst","nhits","nbeats","tft","dlinear","tide","tsmixer","xgboost","arima","farima","theta","seasonal_naive"]
def sm(ds,v,h):
    fs=sorted(glob.glob(f"{P}{ds}__{v}__h{h}__s*.npz"))
    if not fs: return None
    ds_=[np.load(f) for f in fs]
    return ds_[0]["y_true_native"].astype(float), np.mean([d["y_pred_native"].astype(float) for d in ds_],0), len(fs)
def dm(d,h,weights="bartlett"):
    T=len(d); m=d.mean(); x=d-m
    g=lambda k: (x[k:]*x[:T-k]).sum()/T
    v=g(0)
    for k in range(1,h):
        w=(1-k/h) if weights=="bartlett" else 1.0
        v+=2*w*g(k)
    if v<=0: return np.nan,np.nan
    s=m/np.sqrt(v/T)*np.sqrt((T+1-2*h+h*(h-1)/T)/T)
    return s, 2*stats.t.sf(abs(s),T-1)
def holm(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); out=np.empty(m); cur=0
    for r,i in enumerate(o): cur=max(cur,(m-r)*p[i]); out[i]=min(cur,1)
    return out
def bh(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); out=np.empty(m); cur=1
    for r in range(m-1,-1,-1):
        i=o[r]; cur=min(cur,p[i]*m/(r+1)); out[i]=cur
    return out
rows=[]
for ds in DS:
  for h in HS:
    ya,pa,_=sm(ds,"cha_hybrid_v3",h)
    for b in BL:
      r=sm(ds,b,h)
      if r is None: print("missing",ds,h,b); continue
      yb,pb,nb=r; n=min(len(ya),len(yb))
      assert np.allclose(ya[:n],yb[:n])
      e=(ya[:n]-pa[:n])**2-(yb[:n]-pb[:n])**2
      ident=np.max(np.abs(pa[:n]-pb[:n]))<=1e-6*np.std(ya)
      s,p=dm(e.mean(1),h); s2,p2=dm(e.mean(1),h,"rect"); s3,p3=dm(e.ravel(),h)
      rows.append(dict(ds=ds,h=h,b=b,s=s,p=p,s2=s2,p2=p2,s3=s3,p3=p3,ident=ident,nseed=nb))
import pandas as pd
df=pd.DataFrame(rows); ok=~df.ident
print("pairs",len(df),"identical",int(df.ident.sum()),df[df.ident][["ds","h","b"]].values.tolist())
for col,sc in [("p","s"),("p2","s2"),("p3","s3")]:
  for corr in ["raw","bh","holm"]:
    pp=df[col].fillna(1).values.copy()
    if corr!="raw":
        adj=np.full(len(df),np.nan); adj[ok.values]=(bh if corr=="bh" else holm)(pp[ok.values]); pp=adj
    sig=(np.nan_to_num(pp,nan=1)<.05)&ok.values
    bet=int((sig&(df[sc]<0)).sum()); wor=int((sig&(df[sc]>0)).sum())
    wc=df[sig&(df[sc]>0)][["ds","h","b"]].values.tolist()
    print(col,corr,"better",bet,"tie",len(df)-bet-wor,"worse",wor, wc if wor<20 else "")
b=df[df.b=="chronos_bolt_zs"]
print(b[["ds","h","s","p","p2","p3","ident"]].round(4).to_string())
df.to_csv(HERE/"dm_kernel_check.csv",index=False)
