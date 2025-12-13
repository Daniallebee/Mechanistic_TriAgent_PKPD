# --------------------------------------------------------------
# plot_phase2_C.py — PHASE 2 FIGURES (Set C)
# BIS–MAP PHASE-PLANE trajectories for 27 actions
# 120-second horizon
#
# Output directories (MATCHING YOUR TREE):
#
#   plots_phase2/
#       C_plots/
#           C_individual/
#               action_00_phaseplane.png
#               ...
#           C_combined/
#               C_combined_phaseplane.png
#
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from triagent_controller.pas_interface import PASInterface
from triagent_controller.state_action import action_to_infusions, NUM_ACTIONS


# --------------------------------------------------------------
# Directory structure (YOUR STRUCTURE)
# --------------------------------------------------------------

ROOT = "plots_phase2"
C_ROOT = os.path.join(ROOT, "C_plots")
DIR_INDIV = os.path.join(C_ROOT, "C_individual")
DIR_COMBINED = os.path.join(C_ROOT, "C_combined")

os.makedirs(DIR_INDIV, exist_ok=True)
os.makedirs(DIR_COMBINED, exist_ok=True)


# --------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------
HORIZON = 120     # seconds (longer to reveal BIS–MAP dynamics)
DT = 1.0


# --------------------------------------------------------------
# Run full simulation for each action
# --------------------------------------------------------------
def simulate_all_actions():
    results = []
    env = PASInterface(dt=DT)

    for action_id in range(NUM_ACTIONS):
        env.reset()
        inf = action_to_infusions(action_id)

        bis_list = []
        map_list = []

        for t in range(HORIZON):
            obs = env.step(inf)
            bis_list.append(obs["BIS"])
            map_list.append(obs["MAP"])

        results.append({
            "action": action_id,
            "BIS": np.array(bis_list),
            "MAP": np.array(map_list),
        })

    return results


# --------------------------------------------------------------
# Individual phase-plane plot
# --------------------------------------------------------------
def plot_individual(result):
    a = result["action"]
    bis = result["BIS"]
    mapv = result["MAP"]

    plt.figure(figsize=(7, 6))

    # Safe BIS zone 40–60
    plt.axvspan(40, 60, color="lightblue", alpha=0.25)

    # MAP safety zones
    plt.axhspan(65, 200, color="lightgreen", alpha=0.15)   # safe
    plt.axhspan(50, 65, color="yellow", alpha=0.20)        # marginal
    plt.axhspan(0, 50, color="red", alpha=0.20)            # critical

    plt.plot(bis, mapv, "-o", markersize=2, linewidth=1.5, color="black")

    plt.title(f"Action {a:02d} — BIS–MAP Phase Plane (120s)")
    plt.xlabel("BIS")
    plt.ylabel("MAP (mmHg)")
    plt.grid(True)

    fname = os.path.join(DIR_INDIV, f"action_{a:02d}_phaseplane.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------
# Combined 27-panel phase-plane grid
# --------------------------------------------------------------
def plot_combined(results):
    fig, axes = plt.subplots(9, 3, figsize=(18, 30))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        ax = axes[idx]
        bis = result["BIS"]
        mapv = result["MAP"]

        # Safe zones (transparent for readability)
        ax.axvspan(40, 60, color="lightblue", alpha=0.20)
        ax.axhspan(65, 200, color="lightgreen", alpha=0.10)
        ax.axhspan(50, 65, color="yellow", alpha=0.10)
        ax.axhspan(0, 50, color="red", alpha=0.10)

        ax.plot(bis, mapv, color="black", linewidth=1)

        ax.set_title(f"C{idx:02d}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

    plt.tight_layout()
    fname = os.path.join(DIR_COMBINED, "C_combined_phaseplane.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Running PHASE 2 — Set C (Phase-plane BIS–MAP trajectories)...")

    all_results = simulate_all_actions()

    for r in all_results:
        plot_individual(r)

    plot_combined(all_results)

    print("PHASE 2 Set C plots complete.")
    print(f"Saved under: {C_ROOT}")
