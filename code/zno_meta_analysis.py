"""
FINAL pipeline on the merged, corrected dataset (v3: paper 55 merged into 40).
Fills bibliographic metadata, recomputes every statistic, regenerates 7 figures.
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42); np.random.seed(42)

OUT = "../results"; FIG = "../figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
df = pd.read_csv("../data/ZnO_dataset_final.csv")

# ---- bibliographic metadata (per paper); already present in the shipped
#      dataset's DOI / Publication_Year / First_Author columns -- kept here
#      as an inline, human-readable provenance record ----
META = {
 2:("Z. Wu",2016,"10.1039/C6DT02155G"),
 4:("K. G. Kanade",2007,"10.1016/j.matchemphys.2006.11.012"),
 5:("V. Vaiano",2019,"10.3390/app9132741"),
 7:("N. S. Gultom",2017,"10.1016/j.ijhydene.2017.08.135"),
 15:("I. Ahmad",2019,"10.1007/s42452-019-0331-9"),
 23:("P. P. Patel",2015,"10.1016/j.jpowsour.2015.08.027"),
 34:("I. Ahmad",2021,"10.1016/j.ijhydene.2021.05.164"),
 40:("A. B. Patil",2021,"10.1007/s40243-021-00199-5"),
 41:("I. Ahmad",2021,"10.1016/j.jcis.2020.09.132"),
 49:("I. Ahmad",2020,"10.1016/j.mssp.2019.104748"),
 54:("I. Ahmad",2023,"10.1021/acsomega.3c01262"),
}
df["First_Author"]=df["Paper_ID"].map(lambda p: META[p][0])
df["Publication_Year"]=df["Paper_ID"].map(lambda p: META[p][1])
df["DOI"]=df["Paper_ID"].map(lambda p: META[p][2])
# (no resave: data/ZnO_dataset_final.csv already ships with this metadata filled in)

TGT="H2_Rate_umol_h_g"; CONC="Doping_Conc_wt_pct"; PID="Paper_ID"
PER=["Radius_Mismatch","Electronegativity_Mismatch","Charge_Mismatch","Atomic_Weight"]
EXPN=["Doping_Conc_wt_pct","Calcination_Annealing_Temp_C","Reaction_Temp_C","Avg_Wavelength_nm","pH"]
EXPC=["Reaction_Type","Synthesis_Group"]
CORE="Photocatalytic_H2"
R={}
R["n_rows"]=len(df); R["n_papers"]=df[PID].nunique(); R["n_dopants"]=df["Dopant"].nunique()

# ---- effect sizes ----
b1=df[df[CONC]==0].groupby([PID,"Synthesis_Group"])[TGT].mean().rename("b1")
b2=df[df[CONC]==0].groupby(PID)[TGT].mean().rename("b2")
d=df.merge(b1,on=[PID,"Synthesis_Group"],how="left").merge(b2,on=PID,how="left")
d["baseline"]=d["b1"].fillna(d["b2"])
E=d[(d["baseline"].notna())&(d[CONC]>0)].copy()
E["EF"]=E[TGT]/E["baseline"]; E["lnRR"]=np.log(E["EF"])
R.update(n_effects=len(E), n_studies_E=E[PID].nunique(), n_dopants_E=E["Dopant"].nunique())

# ---- REML random intercept ----
def reml(y,g):
    g=pd.factorize(g)[0]; J=g.max()+1; n=len(y)
    def nll(lam):
        A=b=ld=0
        for j in range(J):
            m=g==j; nj=m.sum(); w=1/(1+nj*lam); A+=nj*w; b+=y[m].sum()*w; ld+=np.log(1+nj*lam)
        mu=b/A; r=sum(((y[g==j]-mu)@(y[g==j]-mu))-(lam/(1+(g==j).sum()*lam))*((y[g==j]-mu).sum()**2) for j in range(J))
        return .5*(ld+np.log(A)+(n-1)*np.log(r/(n-1)))
    lam=minimize_scalar(nll,bounds=(1e-9,500),method="bounded").x
    A=b=0
    for j in range(J):
        m=g==j; nj=m.sum(); w=1/(1+nj*lam); A+=nj*w; b+=y[m].sum()*w
    mu=b/A; r=sum(((y[g==j]-mu)@(y[g==j]-mu))-(lam/(1+(g==j).sum()*lam))*((y[g==j]-mu).sum()**2) for j in range(J))
    s2=r/(n-1); t2=lam*s2
    return dict(mu=mu,se=np.sqrt(s2/A),tau2=t2,sigma2=s2,I2=100*t2/(t2+s2))
M=reml(E["lnRR"].values,E[PID].values)
M["EF"]=np.exp(M["mu"]); M["lo"]=np.exp(M["mu"]-1.96*M["se"]); M["hi"]=np.exp(M["mu"]+1.96*M["se"])
M["z"]=M["mu"]/M["se"]; M["p"]=2*(1-stats.norm.cdf(abs(M["z"])))
R["pooled"]={k:float(v) for k,v in M.items()}

# ---- per-dopant bootstrap ----
def boot(sub,B=5000):
    ps=sub[PID].unique()
    if len(ps)==1:
        v=sub["lnRR"].values; bs=[np.mean(RNG.choice(v,len(v),replace=True)) for _ in range(B)]
    else:
        bs=[np.mean([sub.loc[sub[PID]==p,"lnRR"].mean() for p in RNG.choice(ps,len(ps),replace=True)]) for _ in range(B)]
    return np.percentile(bs,[2.5,97.5])
rows=[]
for dop,sub in E.groupby("Dopant"):
    lo,hi=boot(sub)
    rows.append(dict(Dopant=dop,k=len(sub),studies=sub[PID].nunique(),
                     EF=np.exp(sub["lnRR"].mean()),lo=np.exp(lo),hi=np.exp(hi),maxEF=sub["EF"].max()))
FOREST=pd.DataFrame(rows).sort_values("EF",ascending=False).reset_index(drop=True)
R["forest"]=FOREST.to_dict("records")

# ---- design matrices ----
def design(frame):
    Xp=frame[PER].astype(float)
    Xe=pd.concat([frame[EXPN].astype(float),pd.get_dummies(frame[EXPC],drop_first=True).astype(float)],axis=1)
    return Xp.fillna(Xp.median()), Xe.fillna(Xe.median())
Xp_all,Xe_all=design(df); X_all=pd.concat([Xp_all,Xe_all],axis=1)
y_abs=np.log10(df[TGT]+1).values

# ---- leakage: 7 models x 3 schemes ----
MODELS={
 "Random Forest":RandomForestRegressor(n_estimators=800,random_state=42,n_jobs=-1),
 "Extra Trees":ExtraTreesRegressor(n_estimators=800,random_state=42,n_jobs=-1),
 "Gradient Boosting":GradientBoostingRegressor(random_state=42),
 "Hist. Gradient Boosting":HistGradientBoostingRegressor(random_state=42),
 "Ridge (linear)":make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,30))),
 "SVR (RBF)":make_pipeline(StandardScaler(),SVR(C=10,epsilon=0.1)),
 "k-NN (k=5)":make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=5)),
}
SCH={"Random 5-fold":dict(cv=KFold(5,shuffle=True,random_state=42),groups=None),
     "Leave-one-paper-out":dict(cv=LeaveOneGroupOut(),groups=df[PID].values),
     "Leave-one-dopant-out":dict(cv=LeaveOneGroupOut(),groups=df["Dopant"].values)}
tab=[]
for mn,mdl in MODELS.items():
    rec={"Model":mn}
    for sn,kw in SCH.items():
        rec[sn]=r2_score(y_abs,cross_val_predict(mdl,X_all,y_abs,cv=kw["cv"],groups=kw["groups"],n_jobs=-1))
    rec["Leakage gap"]=rec["Random 5-fold"]-rec["Leave-one-paper-out"]; tab.append(rec)
T=pd.DataFrame(tab); T.to_csv(f"{OUT}/leakage_table.csv",index=False)
R["leakage_table"]=T.to_dict("records")
rf_row=T[T["Model"]=="Random Forest"].iloc[0]
R["leakage_rf"]={"random":float(rf_row["Random 5-fold"]),"paper":float(rf_row["Leave-one-paper-out"]),
                 "dopant":float(rf_row["Leave-one-dopant-out"])}
R["leakage_mean_gap"]=float(T["Leakage gap"].mean())
R["leakage_lopo_range"]=[float(T["Leave-one-paper-out"].min()),float(T["Leave-one-paper-out"].max())]
R["leakage_rand_range"]=[float(T["Random 5-fold"].min()),float(T["Random 5-fold"].max())]

# ---- commonality ----
def r2in(X,y):
    if X.shape[1]==0: return 0.0
    m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,30))); m.fit(X,y); return r2_score(y,m.predict(X))
def comm(Xp,Xe,y):
    Rp,Re=r2in(Xp,y),r2in(Xe,y); Rpe=r2in(pd.concat([Xp,Xe],axis=1),y)
    return dict(unique_periodic=Rpe-Re,unique_experimental=Rpe-Rp,shared=Rp+Re-Rpe,total=Rpe)
Xp_E,Xe_E=design(E)
R["comm_abs"]=comm(Xp_all,Xe_all,y_abs)
R["comm_ef"]=comm(Xp_E,Xe_E,E["lnRR"].values)

# ---- moderators ----
desc=df.groupby("Dopant")[PER].first()
lvl=desc.join(E.groupby("Dopant")["lnRR"].mean().rename("lnRR"),how="inner").dropna()
MOD=[]
for c in PER:
    rho,p=stats.spearmanr(lvl[c],lvl["lnRR"])
    bs=[]
    for _ in range(5000):
        i=RNG.choice(len(lvl),len(lvl),replace=True)
        if len(np.unique(lvl[c].values[i]))<3: continue
        bs.append(stats.spearmanr(lvl[c].values[i],lvl["lnRR"].values[i])[0])
    lo,hi=np.nanpercentile(bs,[2.5,97.5]); MOD.append(dict(feature=c,rho=rho,p=p,lo=lo,hi=hi))
MOD=pd.DataFrame(MOD).sort_values("rho",ascending=False); R["moderators"]=MOD.to_dict("records")

# ---- baseline spread ----
BASE=(E.groupby(PID).agg(baseline=("baseline","first"),doped=(TGT,"mean"),
      dopant=("Dopant","first")).sort_values("baseline"))
R["baseline_spread"]=float(BASE["baseline"].max()/BASE["baseline"].min())
R["doped_spread"]=float(BASE["doped"].max()/BASE["doped"].min())
R["baseline_min"]=float(BASE["baseline"].min()); R["baseline_max"]=float(BASE["baseline"].max())

# reaction subgroup
R["subgroup"]={rt:float(np.exp(sub["lnRR"].mean())) for rt,sub in E.groupby("Reaction_Type")}
# photocat-only small-study check
C=E[E["Reaction_Type"]==CORE]
if len(C)>3:
    rho,p=stats.spearmanr(np.log10(C["baseline"]),C["lnRR"]); R["photocat_smallstudy"]=[float(rho),float(p)]

# n single-study dopants
nsingle=int((FOREST["studies"]==1).sum()); R["n_single_dopants"]=nsingle

# ================= FIGURES =================
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi":200,"savefig.dpi":300,"font.size":10,"axes.titlesize":12,
    "axes.titleweight":"bold","axes.labelsize":10.5,"xtick.labelsize":9.5,"ytick.labelsize":9.5,
    "legend.fontsize":9,"axes.edgecolor":"#444444","savefig.bbox":"tight","font.family":"DejaVu Sans"})
BLUE,ORANGE,GREEN,RED,GREY="#4C72B0","#DD8452","#55A868","#C44E52","#9E9E9E"
def ds(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
RXN_LABEL={"Photocatalytic_H2":"Photocatalytic H$_2$","Photoreforming":"Photoreforming",
           "H2S splitting":"H$_2$S splitting","Plasma-Assisted Water Splitting":"Plasma-assisted",
           "PEC Water Splitting":"PEC water splitting"}

# FIG 1
fig,ax=plt.subplots(1,2,figsize=(12.5,5),gridspec_kw={"width_ratios":[1.5,1]})
mat=pd.crosstab(df["Dopant"],df[PID])
im=ax[0].imshow(mat.values,cmap="Blues",aspect="auto",vmin=0,vmax=mat.values.max())
ax[0].set_xticks(range(mat.shape[1])); ax[0].set_xticklabels(mat.columns,fontsize=9)
ax[0].set_yticks(range(mat.shape[0])); ax[0].set_yticklabels(mat.index,fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v=mat.values[i,j]
        if v: ax[0].text(j,i,v,ha="center",va="center",fontsize=8,color="white" if v>mat.values.max()*0.55 else "#222")
ax[0].set_xlabel("Source publication (study ID)"); ax[0].set_ylabel("Dopant")
ax[0].set_title("Each dopant is studied by essentially one laboratory",fontsize=11.5); ax[0].grid(False)
nsingle_all=int((mat>0).sum(axis=1).eq(1).sum()); R["n_single_dopants_all"]=nsingle_all
ax[0].text(0.99,-0.16,f"{nsingle_all} of {mat.shape[0]} dopants appear in a single study",
           transform=ax[0].transAxes,ha="right",fontsize=9,style="italic",color="#555")
cnt=df["Reaction_Type"].value_counts()
ax[1].barh(range(len(cnt)),cnt.values,color=[BLUE if c==CORE else GREY for c in cnt.index])
ax[1].set_yticks(range(len(cnt))); ax[1].set_yticklabels([RXN_LABEL.get(c,c) for c in cnt.index],fontsize=9); ax[1].invert_yaxis()
for i,v in enumerate(cnt.values): ax[1].text(v+0.3,i,str(v),va="center",fontsize=9)
ax[1].set_xlabel("Number of measurements"); ax[1].set_title("Five distinct H$_2$-producing reactions are mixed",fontsize=11.5); ds(ax[1])
fig.suptitle(f"Figure 1 | Structure of the literature-derived dataset "
             f"({R['n_rows']} measurements, {R['n_papers']} studies, {R['n_dopants']} dopants)",fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_dataset_structure.png"); plt.close(fig)

# FIG 2
fig,ax=plt.subplots(figsize=(10,5.6)); x=np.arange(len(BASE))
ax.vlines(x,BASE["baseline"],BASE["doped"],color="#CCC",lw=2,zorder=1)
ax.scatter(x,BASE["baseline"],s=95,color=GREY,edgecolor="#555",zorder=3,label="Undoped ZnO (same-study control)")
ax.scatter(x,BASE["doped"],s=95,color=ORANGE,edgecolor="#8a4a1d",zorder=3,label="Doped ZnO (study mean)")
ax.set_yscale("log"); ax.set_xticks(x)
ax.set_xticklabels([f"S{p}\n{r.dopant}" for p,r in BASE.iterrows()],fontsize=8.5)
ax.set_ylabel("H$_2$ evolution rate  (µmol h$^{-1}$ g$^{-1}$, log scale)"); ax.set_xlabel("Study (sorted by undoped baseline)")
ax.set_title(f"Reported rates for the SAME undoped ZnO span {R['baseline_spread']:,.0f}-fold across studies",fontsize=12.5)
ax.annotate("",xy=(-0.55,BASE["baseline"].min()),xytext=(-0.55,BASE["baseline"].max()),arrowprops=dict(arrowstyle="<->",color=RED,lw=1.8))
ax.text(-0.9,np.sqrt(BASE["baseline"].min()*BASE["baseline"].max()),f"{R['baseline_spread']:,.0f}×",rotation=90,ha="center",va="center",color=RED,fontsize=11,fontweight="bold")
ax.legend(loc="upper left",frameon=True)
ax.text(0.995,0.02,"Absolute rates are not comparable between laboratories;\nwithin-study ratios (enhancement factors) are.",transform=ax.transAxes,ha="right",fontsize=9,style="italic",color="#555")
ds(ax); fig.tight_layout(); fig.savefig(f"{FIG}/fig2_baseline_heterogeneity.png"); plt.close(fig)

# FIG 3 (RF, 3 schemes)
rf=RandomForestRegressor(n_estimators=800,random_state=42,n_jobs=-1)
leak={s:cross_val_predict(rf,X_all,y_abs,cv=SCH[s]["cv"],groups=SCH[s]["groups"]) for s in SCH}
L={s:r2_score(y_abs,leak[s]) for s in SCH}
fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharex=True,sharey=True)
subt=["rows from the same study\nappear in train and test","entire study held out","entire dopant held out"]
cols=[RED,ORANGE,BLUE]
lim=(min(y_abs.min(),min(v.min() for v in leak.values()))-0.3,max(y_abs.max(),max(v.max() for v in leak.values()))+0.3)
for a,s,st,c in zip(axes,SCH,subt,cols):
    a.plot(lim,lim,"--",color="#888",lw=1.2); a.scatter(y_abs,leak[s],s=42,color=c,alpha=0.75,edgecolor="white",linewidth=0.6)
    a.set_xlim(lim); a.set_ylim(lim); a.set_title(f"{s}\nR² = {L[s]:+.2f}",fontsize=11.5)
    a.text(0.04,0.93,st,transform=a.transAxes,fontsize=8.6,va="top",style="italic",color="#555"); a.set_xlabel("Observed  log$_{10}$ H$_2$ rate"); ds(a)
axes[0].set_ylabel("Predicted  log$_{10}$ H$_2$ rate")
fig.suptitle("Figure 3 | Identical model, three validation schemes: apparent accuracy collapses once studies are held out",fontsize=13,y=1.04)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_leakage.png"); plt.close(fig)
R["leakage_rf"]={"random":float(L["Random 5-fold"]),"paper":float(L["Leave-one-paper-out"]),"dopant":float(L["Leave-one-dopant-out"])}

# FIG 3b (multi-algo)
fig,ax=plt.subplots(figsize=(9.2,6.0)); xpos=[0,1,2]
markers=["o","s","^","D","v","P","X"]
for (_,r),mk in zip(T.iterrows(),markers):
    vals=[r["Random 5-fold"],r["Leave-one-paper-out"],r["Leave-one-dopant-out"]]
    ax.plot(xpos,vals,"-",color="#9AA5B1",lw=1.4,alpha=0.85,zorder=1)
    ax.scatter(xpos,vals,marker=mk,s=70,zorder=3,label=r["Model"],edgecolor="white",linewidth=0.8)
ax.axhline(0,color="#666",ls="--",lw=1.1)
ax.text(2.32,0.015,"R² = 0\n(predicting the mean)",fontsize=8.4,color="#666",va="bottom")
mr,ml=T["Random 5-fold"].mean(),T["Leave-one-paper-out"].mean()
ax.annotate("",xy=(0.5,ml+0.02),xytext=(0.5,mr-0.02),arrowprops=dict(arrowstyle="->",color=RED,lw=2))
ax.text(0.55,(mr+ml)/2,f"mean leakage gap\nΔR² = {T['Leakage gap'].mean():+.2f}",color=RED,fontsize=10,fontweight="bold",va="center")
ax.set_xticks(xpos); ax.set_xticklabels(["Random\n5-fold CV","Leave-one-\npaper-out","Leave-one-\ndopant-out"],fontsize=10.5)
ax.set_xlim(-0.35,3.0); ax.set_ylabel("Cross-validated R²  (log₁₀ H₂ rate)")
ax.set_title("Seven model families, one conclusion:\napparent accuracy is a property of the validation scheme, not the algorithm",fontsize=12)
ax.legend(loc="center right",frameon=True,fontsize=8.6,title="Model",title_fontsize=9); ds(ax)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3b_leakage_multialgo.png"); plt.close(fig)

# FIG 4 forest
fig,ax=plt.subplots(figsize=(9.5,6.2)); F=FOREST.iloc[::-1].reset_index(drop=True); yy=np.arange(len(F))
for i,r in F.iterrows():
    c=BLUE if r["studies"]>1 else GREY
    ax.plot([r["lo"],r["hi"]],[i,i],color=c,lw=2.2,zorder=2)
    ax.scatter(r["EF"],i,s=40+14*r["k"],color=c,edgecolor="white",linewidth=0.8,zorder=3)
ax.axvline(1,color="#888",ls="--",lw=1.2); ax.axvline(M["EF"],color=RED,lw=1.6,alpha=0.85); ax.axvspan(M["lo"],M["hi"],color=RED,alpha=0.10)
ax.set_xscale("log"); ax.set_yticks(yy)
ax.set_yticklabels([f"{r.Dopant}   (k={r.k}, {r.studies} stud"+("y" if r.studies==1 else "ies")+")" for r in F.itertuples()],fontsize=9.5)
ax.set_xlabel("Enhancement factor   (doped ÷ undoped, same study; log scale)")
ax.set_title(f"Pooled enhancement = {M['EF']:.1f}×  [95% CI {M['lo']:.1f}–{M['hi']:.1f}],  I² = {M['I2']:.0f}%",fontsize=12)
ax.text(0.99,0.02,"grey = single study (dopant effect confounded with laboratory)\nred line = pooled random-effects estimate",transform=ax.transAxes,ha="right",fontsize=8.8,style="italic",color="#555"); ds(ax)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_forest.png"); plt.close(fig)

# FIG 5 ranking flip
abs_rank=df.groupby("Dopant")[TGT].mean().rank(ascending=False)
ef_rank=E.groupby("Dopant")["lnRR"].mean().rank(ascending=False)
common=sorted(set(abs_rank.index)&set(ef_rank.index))
fig,ax=plt.subplots(figsize=(7.6,6.4))
for dop in common:
    a,b=abs_rank[dop],ef_rank[dop]; mv=b-a
    c=GREEN if mv<-0.5 else (RED if mv>0.5 else GREY)
    ax.plot([0,1],[a,b],"-o",color=c,lw=2.1,ms=7,alpha=0.9)
    ax.text(-0.045,a,dop,ha="right",va="center",fontsize=10,color=c,fontweight="bold")
    ax.text(1.045,b,dop,ha="left",va="center",fontsize=10,color=c,fontweight="bold")
ax.set_xlim(-0.28,1.28); ax.invert_yaxis(); ax.set_xticks([0,1])
ax.set_xticklabels(["Ranked by\nabsolute rate","Ranked by\nenhancement factor"],fontsize=10.5); ax.set_ylabel("Rank  (1 = best)")
ax.set_title("Correcting for the study baseline reverses the dopant ranking",fontsize=12)
ax.legend(handles=[Patch(color=GREEN,label="promoted"),Patch(color=RED,label="demoted"),Patch(color=GREY,label="unchanged")],loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.17))
ax.grid(axis="x",visible=False); ds(ax); fig.tight_layout(); fig.savefig(f"{FIG}/fig5_ranking_flip.png"); plt.close(fig)

# FIG 6
fig,axes=plt.subplots(1,2,figsize=(12.5,5),gridspec_kw={"width_ratios":[1.1,1]})
best=MOD.iloc[0]; xcol=best["feature"]
axes[0].scatter(lvl[xcol],np.exp(lvl["lnRR"]),s=140,color=BLUE,edgecolor="white",linewidth=1,zorder=3)
for dop,r in lvl.iterrows():
    axes[0].annotate(dop,(r[xcol],np.exp(r["lnRR"])),textcoords="offset points",xytext=(7,5),fontsize=10,fontweight="bold",color="#333")
sl,ic=np.polyfit(lvl[xcol],lvl["lnRR"],1); xs=np.linspace(lvl[xcol].min(),lvl[xcol].max(),60)
axes[0].plot(xs,np.exp(ic+sl*xs),"-",color=RED,lw=1.8,alpha=0.85); axes[0].set_yscale("log")
axes[0].set_xlabel("Ionic-radius mismatch  |r$_{dopant}$ − r$_{Zn^{2+}}$|  (Å)"); axes[0].set_ylabel("Mean enhancement factor (log scale)")
axes[0].set_title(f"Radius mismatch is the only significant moderator\nSpearman ρ = {best['rho']:+.2f} (p = {best['p']:.3f}), 95% CI [{best['lo']:+.2f}, {best['hi']:+.2f}]",fontsize=11); ds(axes[0])
labels=["Unique to\nelemental\n(periodic)","Unique to\nexperimental\nconditions","Shared\n(cannot be\nattributed)"]
va=[R["comm_abs"]["unique_periodic"],R["comm_abs"]["unique_experimental"],R["comm_abs"]["shared"]]
ve=[R["comm_ef"]["unique_periodic"],R["comm_ef"]["unique_experimental"],R["comm_ef"]["shared"]]
xp=np.arange(3); w=0.38
axes[1].bar(xp-w/2,va,w,label="Target: absolute rate",color=BLUE); axes[1].bar(xp+w/2,ve,w,label="Target: enhancement factor",color=ORANGE)
for xx,v in zip(xp-w/2,va): axes[1].text(xx,v+0.012,f"{v:.2f}",ha="center",fontsize=9)
for xx,v in zip(xp+w/2,ve): axes[1].text(xx,v+0.012,f"{v:.2f}",ha="center",fontsize=9)
axes[1].set_xticks(xp); axes[1].set_xticklabels(labels,fontsize=9.5); axes[1].set_ylabel("Share of explained variance (R²)")
axes[1].set_title("Most explained variance cannot be attributed\nto chemistry or to conditions separately",fontsize=11); axes[1].legend(frameon=True,loc="upper left"); ds(axes[1])
fig.suptitle("Figure 6 | What drives the enhancement, and how much we can attribute",fontsize=13,y=1.03)
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_moderator_variance.png"); plt.close(fig)

E.to_csv(f"{OUT}/effect_sizes.csv",index=False)
json.dump(R,open(f"{OUT}/results_numbers.json","w"),indent=2,default=float)

print("=== FINAL NUMBERS (merged 11-study dataset) ===")
print(f"rows {R['n_rows']} | papers {R['n_papers']} | dopants {R['n_dopants']}")
print(f"effect sizes {R['n_effects']} | studies {R['n_studies_E']} | dopants {R['n_dopants_E']}")
print(f"pooled EF {M['EF']:.2f} [{M['lo']:.2f},{M['hi']:.2f}] p={M['p']:.2e} I2={M['I2']:.1f}%")
print(f"baseline spread {R['baseline_spread']:.0f}x ({R['baseline_min']:.1f}-{R['baseline_max']:.1f}) | doped {R['doped_spread']:.0f}x")
print(f"leakage RF: rand {L['Random 5-fold']:+.3f} paper {L['Leave-one-paper-out']:+.3f} dopant {L['Leave-one-dopant-out']:+.3f}")
print(f"leakage mean gap {T['Leakage gap'].mean():+.3f} | lopo range {R['leakage_lopo_range']}")
print(f"comm abs shared {R['comm_abs']['shared']:.3f}/{R['comm_abs']['total']:.3f} | ef shared {R['comm_ef']['shared']:.3f}/{R['comm_ef']['total']:.3f}")
print(f"top moderator {best['feature']} rho={best['rho']:+.2f} p={best['p']:.3f}")
print(f"n single-study dopants {nsingle}")
print("\nFOREST:"); print(FOREST.round(2).to_string(index=False))
print("\nMULTI-ALGO:"); print(T.round(3).to_string(index=False))
print("\nSUBGROUP:", {k:round(v,1) for k,v in R["subgroup"].items()})
print("photocat small-study:", R.get("photocat_smallstudy"))
