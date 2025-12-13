# --------------------------------------------------------------
# plot_phase2_B.py — PHASE 2 FIGURES (Set B)
# Effect-site concentration trajectories for 27 actions
# (Ce_prop, Ce_remi, Ce_nore) over a 60-second horizon.
#
# Output structure (YOUR DIRECTORY SETUP):
#   plots_phase2/
#       B_plots/
#           B_individual/
#               action_00_CeTraj.png
#           B_combined/
#               B_combined_CeTraj.png
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from triagent_controller.pas_interface import PASInterface
from triagent_controller.state_action import action_to_infusions, NUM_ACTIONS


# --------------------------------------------------------------
# Directory structure (EXACTLY matching your tree)
# --------------------------------------------------------------
ROOT = "plots_phase2"
B_ROOT = os.path.join(ROOT, "B_plots")
DIR_INDIV = os.path.join(B_ROOT, "B_individual")
DIR_COMBINED = os.path.join(B_ROOT, "B_combined")

os.makedirs(DIR_INDIV, exist_ok=True)
os.makedirs(DIR_COMBINED, exist_ok=True)


# --------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------
HORIZON = 60
DT = 1.0


# --------------------------------------------------------------
# Simulate all actions and collect effect-site concentrations
# --------------------------------------------------------------
def simulate_all_actions():
    results = []
    env = PASInterface(dt=DT)

    for action_id in range(NUM_ACTIONS):
        env.reset()
        inf = action_to_infusions(action_id)

        t_list = []
        ce_prop_list = []
        ce_remi_list = []
        ce_nore_list = []

        for t in range(HORIZON):
            obs = env.step(inf)
            t_list.append(obs["time"])
            ce_prop_list.append(obs["Ce_prop"])
            ce_remi_list.append(obs["Ce_remi"])
            ce_nore_list.append(obs["Ce_nore"])

        results.append({
            "action": action_id,
            "time": np.array(t_list),
            "Ce_prop": np.array(ce_prop_list),
            "Ce_remi": np.array(ce_remi_list),
            "Ce_nore": np.array(ce_nore_list),
        })

    return results


# --------------------------------------------------------------
# Plot INDIVIDUAL effect-site trajectories
# --------------------------------------------------------------
def plot_individual(res):
    a = res["action"]
    t = res["time"]
    ce_p = res["Ce_prop"]
    ce_r = res["Ce_remi"]
    ce_n = res["Ce_nore"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, ce_p, label="Ce_prop", color="blue", linewidth=2)
    plt.plot(t, ce_r, label="Ce_remi", color="green", linewidth=2)
    plt.plot(t, ce_n, label="Ce_nore", color="red", linewidth=2)

    plt.title(f"Action {a:02d} — Effect-Site Concentrations (60s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration")
    plt.grid(True)
    plt.legend()

    fname = os.path.join(DIR_INDIV, f"action_{a:02d}_CeTraj.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------
# Plot COMBINED 27-panel Ce trajectory grid
# --------------------------------------------------------------
def plot_combined(all_results):
    fig, axes = plt.subplots(9, 3, figsize=(18, 30))
    axes = axes.flatten()

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        t = result["time"]
        ce_p = result["Ce_prop"]
        ce_r = result["Ce_remi"]
        ce_n = result["Ce_nore"]

        ax.plot(t, ce_p, color="blue", linewidth=1)
        ax.plot(t, ce_r, color="green", linewidth=1)
        ax.plot(t, ce_n, color="red", linewidth=1)

        ax.set_title(f"B{idx:02d}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

    plt.tight_layout()
    fname = os.path.join(DIR_COMBINED, "B_combined_CeTraj.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Running PHASE 2 — Set B (Effect-site trajectories)...")

    all_res = simulate_all_actions()

    for r in all_res:
        plot_individual(r)

    plot_combined(all_res)

    print("PHASE 2 Set B plots complete.")
    print(f"Saved under: {B_ROOT}")
