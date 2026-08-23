"""
Figure 1 | The validation-aware framework (workflow schematic).
Figure 2 | Structural confounding in the dataset (study-dopant bipartite
           network + information-theoretic confounding metric) and the
           distribution of measurements across reaction types.

Input:  ZnO_dataset_final.csv
Output: fig1_workflow.png, fig2_confounding_network.png
"""
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from scipy.stats import entropy
import os

DATA_PATH = "../data/ZnO_dataset_final.csv"
FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
TARGET_COL = "H2_Rate_umol_h_g"
PAPER_COL = "Paper_ID"
DOPANT_COL = "Dopant"
REACTION_COL = "Reaction_Type"
CORE_REACTION = "Photocatalytic_H2"

plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300, "font.family": "DejaVu Sans", "savefig.bbox": "tight"})
ACCENT = "#1F4E79"; BLUE = "#4C72B0"; ORANGE = "#DD8452"; GREEN = "#55A868"; RED = "#C44E52"; GREY = "#9E9E9E"


# ============================================================ FIGURE 1 ====
def make_workflow_figure(output_path=f"{FIG_DIR}/fig1_workflow.png"):
    """Six-step schematic of the validation-aware meta-analytic framework."""
    steps = [
        ("1. Literature curation", "single-cation-doped ZnO;\nsame-study undoped control required", BLUE),
        ("2. Within-study normalization", "enhancement factor\nEF = doped rate \u00f7 same-study control", GREEN),
        ("3. Multilevel meta-analysis", "random-effects model;\npooled effect + heterogeneity (I\u00b2)", BLUE),
        ("4. Validation ladder", "random vs leave-one-study-out\nvs leave-one-dopant-out \u00d7 7 models", ORANGE),
        ("5. Variance partitioning", "commonality analysis;\nstudy\u2013dopant confounding metric", BLUE),
        ("6. Reporting recommendations", "study-level holdout + baselines\nfor any literature-mined dataset", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 9.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, len(steps) * 2 + 0.5)
    box_w, box_h = 8.4, 1.45
    for i, (title, sub, col) in enumerate(steps):
        y = (len(steps) - 1 - i) * 2 + 0.6
        ax.add_patch(FancyBboxPatch((0.8, y), box_w, box_h,
                     boxstyle="round,pad=0.06,rounding_size=0.12",
                     linewidth=1.6, edgecolor=col, facecolor=col + "22"))
        ax.text(5.0, y + box_h - 0.42, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ACCENT)
        ax.text(5.0, y + 0.36, sub, ha="center", va="center", fontsize=9.3, color="#333333")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y - 0.55), arrowstyle="-|>",
                         mutation_scale=20, linewidth=2, color="#888888"))
    ax.text(0.1, len(steps) * 2 * 0.5 + 0.3,
            "A validation-aware framework\nfor literature-derived catalyst data",
            rotation=90, ha="center", va="center", fontsize=11, fontweight="bold", color=ACCENT)
    fig.tight_layout(); fig.savefig(output_path); plt.close(fig)
    print(f"Saved {output_path}")


# ============================================================ FIGURE 2 ====
def compute_confounding_metrics(df: pd.DataFrame) -> dict:
    """Information-theoretic confounding between dopant and study identity."""
    dopant, study = df[DOPANT_COL].values, df[PAPER_COL].astype(str).values
    mutual_info = mutual_info_score(dopant, study)

    def shannon_entropy(x):
        _, counts = np.unique(x, return_counts=True)
        return entropy(counts / counts.sum())

    h_dopant = shannon_entropy(dopant)
    return dict(
        U=mutual_info / h_dopant,  # uncertainty coefficient U(dopant | study)
        NMI=normalized_mutual_info_score(dopant, study),
    )


def make_confounding_figure(df: pd.DataFrame, output_path=f"{FIG_DIR}/fig2_confounding_network.png"):
    """Bipartite study-dopant network (left) + reaction-type histogram (right)."""
    metrics = compute_confounding_metrics(df)
    study_dopant_matrix = pd.crosstab(df[DOPANT_COL], df[PAPER_COL])
    studies, dopants = list(study_dopant_matrix.columns), list(study_dopant_matrix.index)
    n_single_study = int((study_dopant_matrix > 0).sum(axis=1).eq(1).sum())

    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1])
    ax_net, ax_rxn = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---- bipartite layout: studies on the left, dopants on the right ----
    study_y = {s: y for s, y in zip(studies, np.linspace(0, 1, len(studies)))}
    dopant_y = {d: y for d, y in zip(dopants, np.linspace(0, 1, len(dopants)))}
    x_study, x_dopant = 0.0, 1.0

    for dopant in dopants:
        degree = int((study_dopant_matrix.loc[dopant] > 0).sum())
        for study in studies:
            if study_dopant_matrix.loc[dopant, study] > 0:
                color = GREY if degree == 1 else RED
                ax_net.plot([x_study, x_dopant], [study_y[study], dopant_y[dopant]],
                           color=color, lw=1.1 if degree == 1 else 2.2,
                           alpha=0.55 if degree == 1 else 0.9, zorder=1)

    for study in studies:
        ax_net.scatter(x_study, study_y[study], s=120, color=ACCENT, edgecolor="white", zorder=3)
        ax_net.text(x_study - 0.05, study_y[study], f"S{study}", ha="right", va="center",
                    fontsize=8.5, color=ACCENT)
    for dopant in dopants:
        n_measurements = int(study_dopant_matrix.loc[dopant].sum())
        degree = int((study_dopant_matrix.loc[dopant] > 0).sum())
        color = GREY if degree == 1 else GREEN
        ax_net.scatter(x_dopant, dopant_y[dopant], s=60 + 22 * n_measurements,
                       color=color, edgecolor="white", zorder=3)
        ax_net.text(x_dopant + 0.05, dopant_y[dopant], dopant, ha="left", va="center",
                    fontsize=9.5, fontweight="bold", color="#333333")

    ax_net.set_xlim(-0.35, 1.4); ax_net.set_ylim(-0.08, 1.08); ax_net.axis("off")
    ax_net.text(x_study, 1.11, "Studies", ha="center", fontsize=11, fontweight="bold", color=ACCENT)
    ax_net.text(x_dopant, 1.11, "Dopants", ha="center", fontsize=11, fontweight="bold", color="#333333")
    ax_net.set_title("Study\u2013dopant bipartite structure", fontsize=12, fontweight="bold")
    ax_net.text(0.5, -0.05,
        f"Uncertainty coefficient U(dopant | study) = {metrics['U']:.2f}   (NMI = {metrics['NMI']:.2f})\n"
        f"{n_single_study} of {len(dopants)} dopants link to a single study \u2014 grey; green = studied twice",
        transform=ax_net.transAxes, ha="center", va="top", fontsize=9, style="italic", color="#555555")

    # ---- reaction-type panel ----
    reaction_labels = {
        "Photocatalytic_H2": "Photocatalytic H$_2$", "Photoreforming": "Photoreforming",
        "H2S splitting": "H$_2$S splitting", "Plasma-Assisted Water Splitting": "Plasma-assisted",
        "PEC Water Splitting": "PEC",
    }
    counts = df[REACTION_COL].value_counts()
    ax_rxn.barh(range(len(counts)), counts.values,
               color=[BLUE if c == CORE_REACTION else GREY for c in counts.index])
    ax_rxn.set_yticks(range(len(counts)))
    ax_rxn.set_yticklabels([reaction_labels.get(c, c) for c in counts.index], fontsize=9.5)
    ax_rxn.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax_rxn.text(v + 0.3, i, str(v), va="center", fontsize=9)
    ax_rxn.set_xlabel("Number of measurements")
    ax_rxn.set_title("Five distinct H$_2$-producing reactions", fontsize=12, fontweight="bold")
    ax_rxn.spines["top"].set_visible(False); ax_rxn.spines["right"].set_visible(False)

    fig.suptitle(
        f"Figure 2 | Structural confounding in the literature-derived dataset "
        f"({len(df)} measurements, {len(studies)} studies, {len(dopants)} dopants)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout(); fig.savefig(output_path); plt.close(fig)
    print(f"Saved {output_path}  (U={metrics['U']:.3f}, NMI={metrics['NMI']:.3f})")


# =============================================================== MAIN =====
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    make_workflow_figure()
    make_confounding_figure(df)
