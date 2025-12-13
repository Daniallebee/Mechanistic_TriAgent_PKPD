# --------------------------------------------------------------
# plot_phase2_E.py — PHASE 2 FIGURES (Set E)
# Heatmaps for fixed norepinephrine levels:
#   • BIS at 60 seconds
#   • MAP at 60 seconds
#   • Safety classification
#
# Output:
#   plots_phase2/E_heatmaps/NE0/*.png
#   plots_phase2/E_heatmaps/NE1/*.png
#   plots_phase2/E_heatmaps/NE2/*.png
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from triagent_controller.pas_interface import PASInterface
from triagent_controller.state_action import action_to_infusions


# --------------------------------------------------------------
# Directory structure
# --------------------------------------------------------------
ROOT = "plots_phase2"
DIR_E = os.path.join(ROOT, "E_heatmaps")
os.makedirs(DIR_E, exist_ok=True)

for i in range(3):
    os.makedirs(os.path.join(DIR_E, f"NE{i}"), exist_ok=True)


# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
DT = 1.0
HORIZON = 60     # seconds (consistent with previous sets)


# --------------------------------------------------------------
# Helper: simulate a single action for 60s
# --------------------------------------------------------------
def simulate_action(prop_level, remi_level, nore_level):

    # convert triple → action ID
    action_id = prop_level * 9 + remi_level * 3 + nore_level
    inf = action_to_infusions(action_id)

    env = PASInterface(dt=DT)
    env.reset()

    obs = None
    for _ in range(HORIZON):
        obs = env.step(inf)

    BIS = obs["BIS"]
    MAP = obs["MAP"]

    # classify safety
    if MAP < 50:
        safety_code = 2   # critical
    elif MAP < 65:
        safety_code = 1   # marginal
    else:
        safety_code = 0   # safe

    return BIS, MAP, safety_code


# --------------------------------------------------------------
# Generate heatmaps for each fixed norepinephrine level
# --------------------------------------------------------------
def generate_heatmaps_for_NE(ne_level):

    BIS_map = np.zeros((3, 3))
    MAP_map = np.zeros((3, 3))
    SAF_map = np.zeros((3, 3))

    for p in range(3):
        for r in range(3):
            BIS, MAP, SAF = simulate_action(p, r, ne_level)
            BIS_map[p, r] = BIS
            MAP_map[p, r] = MAP
            SAF_map[p, r] = SAF

    save_path = os.path.join(DIR_E, f"NE{ne_level}")

    # ---- Plot BIS ----
    plt.figure(figsize=(6, 5))
    plt.imshow(BIS_map, cmap="viridis", origin="lower")
    plt.colorbar(label="BIS")
    plt.title(f"BIS Heatmap — NE Level {ne_level}")
    plt.xticks([0,1,2], ["R0","R1","R2"])
    plt.yticks([0,1,2], ["P0","P1","P2"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "BIS_heatmap.png"), dpi=300)
    plt.close()

    # ---- Plot MAP ----
    plt.figure(figsize=(6, 5))
    plt.imshow(MAP_map, cmap="plasma", origin="lower")
    plt.colorbar(label="MAP (mmHg)")
    plt.title(f"MAP Heatmap — NE Level {ne_level}")
    plt.xticks([0,1,2], ["R0","R1","R2"])
    plt.yticks([0,1,2], ["P0","P1","P2"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "MAP_heatmap.png"), dpi=300)
    plt.close()

    # ---- Safety matrix ----
    plt.figure(figsize=(6, 5))
    plt.imshow(SAF_map, cmap="coolwarm", origin="lower", vmin=0, vmax=2)
    plt.colorbar(label="Safety (0=safe, 1=marginal, 2=critical)")
    plt.title(f"Safety Classification — NE Level {ne_level}")
    plt.xticks([0,1,2], ["R0","R1","R2"])
    plt.yticks([0,1,2], ["P0","P1","P2"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "SAFETY_heatmap.png"), dpi=300)
    plt.close()

    print(f"Completed NE={ne_level}: saved to {save_path}")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Running PHASE 2 — Set E (Heatmaps)...")

    for ne in range(3):
        generate_heatmaps_for_NE(ne)

    print("PHASE 2 Set E complete.")
    print(f"Output saved under: {DIR_E}")
