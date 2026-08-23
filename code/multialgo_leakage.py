"""
Multi-algorithm leakage ladder.
7 model families x 3 validation schemes on identical features/target.
Shows the random-CV -> grouped-CV collapse is algorithm-independent.
Outputs: fig3b_leakage_multialgo.png + leakage_table.csv + console table.
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
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
np.random.seed(42)

OUT = "../results"; FIG = "../figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
TGT = "H2_Rate_umol_h_g"
df = pd.read_csv("../data/ZnO_dataset_final.csv")
df["Paper"] = df["Paper_ID"]

PER = ["Radius_Mismatch", "Electronegativity_Mismatch", "Charge_Mismatch", "Atomic_Weight"]
EXPN = ["Doping_Conc_wt_pct", "Calcination_Annealing_Temp_C", "Reaction_Temp_C", "Avg_Wavelength_nm", "pH"]
EXPC = ["Reaction_Type", "Synthesis_Group"]
X = pd.concat([df[PER].astype(float),
               df[EXPN].astype(float),
               pd.get_dummies(df[EXPC], drop_first=True).astype(float)], axis=1)
X = X.fillna(X.median())
y = np.log10(df[TGT] + 1).values

MODELS = {
    "Random Forest":            RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1),
    "Extra Trees":              ExtraTreesRegressor(n_estimators=800, random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingRegressor(random_state=42),
    "Hist. Gradient Boosting":  HistGradientBoostingRegressor(random_state=42),  # LightGBM-style
    "Ridge (linear)":           make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))),
    "SVR (RBF)":                make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
    "k-NN (k=5)":               make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
}
SCHEMES = {
    "Random 5-fold":        dict(cv=KFold(5, shuffle=True, random_state=42), groups=None),
    "Leave-one-paper-out":  dict(cv=LeaveOneGroupOut(), groups=df["Paper"].values),
    "Leave-one-dopant-out": dict(cv=LeaveOneGroupOut(), groups=df["Dopant"].values),
}

rows = []
for mname, model in MODELS.items():
    rec = {"Model": mname}
    for sname, kw in SCHEMES.items():
        pred = cross_val_predict(model, X, y, cv=kw["cv"], groups=kw["groups"], n_jobs=-1)
        rec[sname] = r2_score(y, pred)
    rec["Leakage gap"] = rec["Random 5-fold"] - rec["Leave-one-paper-out"]
    rows.append(rec)
T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/leakage_table.csv", index=False)

print(T.round(3).to_string(index=False))
print()
print(f"Mean leakage gap  : {T['Leakage gap'].mean():+.3f} R2")
print(f"Range random CV   : {T['Random 5-fold'].min():+.2f} .. {T['Random 5-fold'].max():+.2f}")
print(f"Range paper-out   : {T['Leave-one-paper-out'].min():+.2f} .. {T['Leave-one-paper-out'].max():+.2f}")

# ---------------------------------------------------------------- figure --
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300, "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold",
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
RED, ORANGE, BLUE = "#C44E52", "#DD8452", "#4C72B0"

fig, ax = plt.subplots(figsize=(9.2, 6.0))
xpos = [0, 1, 2]
xlabels = ["Random\n5-fold CV", "Leave-one-\npaper-out", "Leave-one-\ndopant-out"]
markers = ["o", "s", "^", "D", "v", "P", "X"]
for (i, r), mk in zip(T.iterrows(), markers):
    vals = [r["Random 5-fold"], r["Leave-one-paper-out"], r["Leave-one-dopant-out"]]
    ax.plot(xpos, vals, "-", color="#9AA5B1", lw=1.4, alpha=0.85, zorder=1)
    ax.scatter(xpos, vals, marker=mk, s=70, zorder=3, label=r["Model"],
               edgecolor="white", linewidth=0.8)
ax.axhline(0, color="#666666", ls="--", lw=1.1)
ax.text(2.32, 0.015, "R² = 0\n(predicting the mean)", fontsize=8.4, color="#666666", va="bottom")

mean_rand = T["Random 5-fold"].mean(); mean_lopo = T["Leave-one-paper-out"].mean()
ax.annotate("", xy=(0.5, mean_lopo + 0.02), xytext=(0.5, mean_rand - 0.02),
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.text(0.55, (mean_rand + mean_lopo) / 2, f"mean leakage gap\nΔR² = {T['Leakage gap'].mean():+.2f}",
        color=RED, fontsize=10, fontweight="bold", va="center")

ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=10.5)
ax.set_xlim(-0.35, 3.0)
ax.set_ylabel("Cross-validated R²  (log₁₀ H₂ rate)")
ax.set_title("Seven model families, one conclusion:\napparent accuracy is a property of the validation scheme, not the algorithm",
             fontsize=12)
ax.legend(loc="center right", frameon=True, fontsize=8.6, title="Model", title_fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3b_leakage_multialgo.png")
print("\nFigure written:", f"{FIG}/fig3b_leakage_multialgo.png")
