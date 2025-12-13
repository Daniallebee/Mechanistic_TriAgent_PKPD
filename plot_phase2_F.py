# --------------------------------------------------------------
# plot_phase2_F.py — PHASE 2 MASTER FIGURE ASSEMBLY (Set F)
#
# Fully corrected to match YOUR folder structure:
#
#   plots_phase2/
#       A_plots/A_combined/
#       B_plots/B_combined/
#       C_plots/C_combined/
#       E_heatmaps/NE1/
#       F_master/
#
# --------------------------------------------------------------

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg

# --------------------------------------------------------------
# Directory structure (YOUR REAL TREE)
# --------------------------------------------------------------

ROOT = "plots_phase2"

A_COMBINED = os.path.join(ROOT, "A_plots", "A_combined", "A_combined_BIS_MAP.png")
B_COMBINED = os.path.join(ROOT, "B_plots", "B_combined", "B_combined_CeTraj.png")
C_COMBINED = os.path.join(ROOT, "C_plots", "C_combined", "C_combined_phaseplane.png")

E_NE1 = os.path.join(ROOT, "E_heatmaps", "NE1")

# Output directory
DIR_F = os.path.join(ROOT, "F_master")
os.makedirs(DIR_F, exist_ok=True)


# --------------------------------------------------------------
# Helper to load an image safely
# --------------------------------------------------------------
def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMAGE NOT FOUND: {path}")
    return mpimg.imread(path)


# --------------------------------------------------------------
# FIGURE F1 — A+B+C Combined System Response
# --------------------------------------------------------------
def make_F1():

    imgA = load(A_COMBINED)
    imgB = load(B_COMBINED)
    imgC = load(C_COMBINED)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(imgA)
    ax1.axis("off")
    ax1.set_title("A — BIS & MAP Trajectories")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(imgB)
    ax2.axis("off")
    ax2.set_title("B — Effect-Site Concentrations")

    ax3 = fig.add_subplot(gs[1, :])
    ax3.imshow(imgC)
    ax3.axis("off")
    ax3.set_title("C — BIS–MAP Phase Plane")

    fig.tight_layout()
    fname = os.path.join(DIR_F, "Figure_F1_system_response.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved F1 → {fname}")


# --------------------------------------------------------------
# FIGURE F2 — Heatmaps for NE = 1
# --------------------------------------------------------------
def make_F2():

    path_BIS = os.path.join(E_NE1, "BIS_heatmap.png")
    path_MAP = os.path.join(E_NE1, "MAP_heatmap.png")
    path_SAF = os.path.join(E_NE1, "SAFETY_heatmap.png")

    imgBIS = load(path_BIS)
    imgMAP = load(path_MAP)
    imgSAF = load(path_SAF)

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3)

    for idx, (img, title) in enumerate([
        (imgBIS, "BIS Heatmap (NE=1)"),
        (imgMAP, "MAP Heatmap (NE=1)"),
        (imgSAF, "Safety Heatmap (NE=1)")
    ]):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)

    fig.tight_layout()
    fname = os.path.join(DIR_F, "Figure_F2_heatmaps_NE1.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved F2 → {fname}")


# --------------------------------------------------------------
# FIGURE F3 — Best Action Highlight (Action 13)
# --------------------------------------------------------------
def make_F3():
    best_path = os.path.join(ROOT, "A_plots", "A_individual", "action_13_BIS_MAP.png")
    img = load(best_path)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Best Action Trajectory (Action 13)")

    fname = os.path.join(DIR_F, "Figure_F3_best_action.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved F3 → {fname}")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Assembling PHASE 2 — MASTER FIGURES (Set F)...")

    make_F1()
    make_F2()
    make_F3()

    print("\nALL Set F figures successfully generated.")
    print(f"Saved under: {DIR_F}\n")
