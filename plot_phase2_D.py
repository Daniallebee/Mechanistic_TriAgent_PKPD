# --------------------------------------------------------------
# plot_phase2_D.py — PHASE 2 FIGURES (Set D)
# 3D response surfaces for BIS and MAP after 60-second horizon
#
# Output:
#   plots_phase2/D_surfaces/BIS_surface.png
#   plots_phase2/D_surfaces/MAP_surface.png
#   + optional second-angle views
#
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from triagent_controller.pas_interface import PASInterface
from triagent_controller.state_action import action_to_infusions, NUM_ACTIONS


# --------------------------------------------------------------
# Folder setup
# --------------------------------------------------------------
ROOT = "plots_phase2"
DIR_SURF = os.path.join(ROOT, "D_surfaces")
os.makedirs(DIR_SURF, exist_ok=True)


# --------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------
DT = 1.0
HORIZON = 60   # seconds (consistent with Sets A–C)


# --------------------------------------------------------------
# Build 3D matrices: BIS[prop, remi], MAP[remi, nore]
# --------------------------------------------------------------
def collect_surfaces():

    BIS_surface = np.zeros((3, 3))   # prop × remi
    MAP_surface = np.zeros((3, 3))   # remi × nore

    # We need to iterate all 27 actions
    for action_id in range(NUM_ACTIONS):

        p, r, n = action_to_prn(action_id)

        env = PASInterface(dt=DT)
        env.reset()
        inf = action_to_infusions(action_id)

        OBS = None
        for t in range(HORIZON):
            OBS = env.step(inf)

        # After 60 seconds, extract steady-state-ish outputs
        bis_val = OBS["BIS"]
        map_val = OBS["MAP"]

        # Fill BIS surface for prop–remi plane (holding nore fixed implicitly)
        BIS_surface[p, r] = bis_val

        # Fill MAP surface for remi–nore plane
        MAP_surface[r, n] = map_val

    return BIS_surface, MAP_surface


# --------------------------------------------------------------
# Helper: decode (p,r,n) from action ID
# --------------------------------------------------------------
def action_to_prn(action_id: int):
    """
    ACTIONS = (p, r, n) with p,r,n ∈ {0,1,2}
    3 levels × 3 levels × 3 levels = 27 actions
    """
    p = action_id // 9
    r = (action_id % 9) // 3
    n = action_id % 3
    return p, r, n


# --------------------------------------------------------------
# 3D Plotting helper
# --------------------------------------------------------------
def plot_surface(Z, x_labels, y_labels, title, filename):
    X, Y = np.meshgrid(np.arange(3), np.arange(3))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.5,
        antialiased=True,
    )

    ax.set_xticks([0,1,2])
    ax.set_yticks([0,1,2])

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Level")
    ax.set_ylabel("Level")
    ax.set_zlabel("Value")

    fig.colorbar(surf, shrink=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(DIR_SURF, filename), dpi=300)
    plt.close()

    # Second angle (optional but useful)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.5,
        antialiased=True,
    )
    ax.view_init(elev=35, azim=45)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_title(title + " (Angle 2)", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_SURF, "angle2_" + filename), dpi=300)
    plt.close()


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Running PHASE 2 — Set D (3D surfaces)...")

    BIS_surface, MAP_surface = collect_surfaces()

    print("Plotting BIS surface...")
    plot_surface(
        BIS_surface,
        x_labels=["P0","P1","P2"],
        y_labels=["R0","R1","R2"],
        title="BIS Surface Over Propofol × Remifentanil (60s)",
        filename="BIS_surface.png",
    )

    print("Plotting MAP surface...")
    plot_surface(
        MAP_surface,
        x_labels=["R0","R1","R2"],
        y_labels=["N0","N1","N2"],
        title="MAP Surface Over Remifentanil × Norepinephrine (60s)",
        filename="MAP_surface.png",
    )

    print("PHASE 2 Set D complete.")
    print(f"Saved to: {DIR_SURF}")
