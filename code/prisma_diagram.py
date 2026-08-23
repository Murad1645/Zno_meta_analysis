"""
Simplified PRISMA 2020 flow diagram - Figure S1 (v2).
Reflects the actual workflow: a single combined literature search, all
candidate papers downloaded and read in full (no separate title/abstract
screening stage was tracked), then eligibility and meta-analysis filtering.
All numbers below are fixed by the author-confirmed counts (60 read, 11
included) and the final dataset (10 studies / 43 effect sizes / 9 dopants
in the meta-analysis; 1 study excluded for lacking an undoped control).
"""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import os
OUT = "../figures/figS1_prisma.png"
os.makedirs("../figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300, "font.family": "DejaVu Sans", "savefig.bbox": "tight"})
ACCENT = "#1F4E79"; BLUE = "#4C72B0"; GREEN = "#55A868"; GREY = "#666666"; EXC = "#F2F2F2"; LIGHT = "#EAF0F7"

fig, ax = plt.subplots(figsize=(9.6, 8.6)); ax.axis("off")
ax.set_xlim(0, 12); ax.set_ylim(0, 11)

def phase(y, h, txt, color=ACCENT):
    ax.add_patch(FancyBboxPatch((0.05, y-h/2), 0.72, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                 facecolor=color, edgecolor="none"))
    ax.text(0.41, y, txt, rotation=90, ha="center", va="center", fontsize=11, color="white", fontweight="bold")

def box(x, y, w, h, txt, fill=LIGHT, edge=BLUE, fs=10.2, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y-h/2), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                 facecolor=fill, edgecolor=edge, linewidth=1.5))
    lines = txt.split("\n")
    n = len(lines); step = (h-0.28) / max(n, 1)
    y0 = y + (n-1)*step/2
    for i, ln in enumerate(lines):
        ax.text(x+w/2, y0 - i*step, ln, ha="center", va="center",
                fontsize=fs, color="#222", fontweight="bold" if (bold_first and i == 0) else "normal")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=GREY))

ax.text(6.0, 10.5, "Figure S1 | Study selection (simplified PRISMA 2020)", ha="center", fontsize=13.5, fontweight="bold", color=ACCENT)

# ---------------- Identification ----------------
phase(8.9, 1.7, "Identification", BLUE)
box(1.4, 8.9, 6.4, 1.5,
    "Records identified through a single combined\nliterature and citation search\n(n \u2248 60)", fs=10.5)

# ---------------- Eligibility ----------------
phase(6.6, 2.0, "Eligibility", ACCENT)
box(1.4, 6.6, 6.4, 1.6, "Full-text articles downloaded\nand assessed for eligibility\n(n = 60)", fs=10.5)
arrow(4.6, 8.05, 4.6, 7.45)
box(8.6, 6.6, 3.2, 2.1,
    "Full-text articles excluded:\n\u2022 no single-cation ZnO\n\u2022 no H\u2082 rate in \u00b5mol h\u207b\u00b9 g\u207b\u00b9\n\u2022 no same-study undoped control\n(n = 49)",
    fill=EXC, edge=GREY, fs=8.7)
arrow(7.8, 6.6, 8.6, 6.6)

# ---------------- Included ----------------
phase(3.7, 3.4, "Included", GREEN)
box(1.4, 4.3, 6.4, 1.3, "Studies included in\nqualitative synthesis\n(n = 11)", fill="#E4EFE4", edge=GREEN, fs=10.5)
arrow(4.6, 5.8, 4.6, 4.95)

box(1.4, 2.2, 6.4, 1.7,
    "Studies included in\nquantitative meta-analysis\n(n = 10; 43 effect sizes,\n9 dopants)", fill="#E4EFE4", edge=GREEN, fs=10)
arrow(4.6, 3.65, 4.6, 3.05)
box(8.6, 2.2, 3.2, 1.5,
    "Excluded from meta-analysis:\nno undoped control\n(n = 1; Lu study)", fill=EXC, edge=GREY, fs=9)
arrow(7.8, 2.2, 8.6, 2.2)

ax.text(6.0, 0.35,
        "Title/abstract screening and full-text assessment were performed together as a single pass over the\n"
        "combined search results, so they are reported as one stage; duplicates were not tracked separately.\n"
        "Included-study counts (11 qualitative; 10 studies / 43 effect sizes / 9 dopants quantitative) are fixed by\n"
        "the final dataset and match Supplementary Table S1.",
        ha="center", va="center", fontsize=8.5, style="italic", color="#555")

fig.tight_layout(); fig.savefig(OUT); plt.close(fig)
print("saved", OUT)
