# --------------------------------------------------------------
# plot_phase2_A.py — PHASE 2 FIGURES (Set A)
# BIS + MAP trajectories for 27 actions (60-second horizon)
#
# Output structure (matching YOUR folder design):
#   plots_phase2/
#       A_plots/
#           A_individual/
#               action_00_BIS_MAP.png
#               ...
#           A_combined/
#               A_combined_BIS_MAP.png
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from triagent_controller.pas_interface import PASInterface
from triagent_controller.state_action import action_to_infusions, NUM_ACTIONS


# --------------------------------------------------------------
# Directory structure (MATCHES YOUR EXACT ORGANIZATION)
# --------------------------------------------------------------
ROOT = "plots_phase2"
A_ROOT = os.path.join(ROOT, "A_plots")
DIR_INDIV = os.path.join(A_ROOT, "A_individual")
DIR_COMBINED = os.path.join(A_ROOT, "A_combined")

os.makedirs(DIR_INDIV, exist_ok=True)
os.makedirs(DIR_COMBINED, exist_ok=True)


# --------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------
HORIZON = 60
DT = 1.0


# --------------------------------------------------------------
# Run simulation for all 27 actions
# --------------------------------------------------------------
def simulate_all_actions():
    all_results = []

    env = PASInterface(dt=DT)

    for action_id in range(NUM_ACTIONS):
        env.reset()
        inf = action_to_infusions(action_id)

        t_list = []
        bis_list = []
        map_list = []

        for t in range(HORIZON):
            obs = env.step(inf)
            t_list.append(obs["time"])
            bis_list.append(obs["BIS"])
            map_list.append(obs["MAP"])

        all_results.append({
            "action": action_id,
            "time": np.array(t_list),
            "BIS": np.array(bis_list),
            "MAP": np.array(map_list),
        })

    return all_results


# --------------------------------------------------------------
# Individual plots
# --------------------------------------------------------------
def plot_individual(result):
    a = result["action"]
    t = result["time"]
    bis = result["BIS"]
    mapv = result["MAP"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, bis, label="BIS", color="blue", linewidth=2)
    plt.plot(t, mapv, label="MAP (mmHg)", color="red", linewidth=2)

    plt.title(f"Action {a:02d} — BIS & MAP Trajectory (60s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    fname = os.path.join(DIR_INDIV, f"action_{a:02d}_BIS_MAP.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------
# Combined 27-panel plot
# --------------------------------------------------------------
def plot_combined(all_results):
    fig, axes = plt.subplots(9, 3, figsize=(18, 30))
    axes = axes.flatten()

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        t = result["time"]
        bis = result["BIS"]
        mapv = result["MAP"]

        ax.plot(t, bis, color="blue", linewidth=1.2)
        ax.plot(t, mapv, color="red", linewidth=1.2)
        ax.set_title(f"A{idx:02d}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

    plt.tight_layout()
    fname = os.path.join(DIR_COMBINED, "A_combined_BIS_MAP.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Running PHASE 2 — Set A (BIS/MAP trajectories)...")

    results = simulate_all_actions()

    for res in results:
        plot_individual(res)

    plot_combined(results)

    print("PHASE 2 Set A plots complete.")
    print(f"Saved under: {A_ROOT}")
