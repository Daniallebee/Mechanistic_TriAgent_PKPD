# --------------------------------------------------------------
# phase2_test.py — FINAL PHASE 2 DIAGNOSTIC HARNESS
#
# This file:
#   ✓ Creates PASInterface (Eleveld PK, Su-model hemo, BIS PD)
#   ✓ Iterates through ALL 27 actions
#   ✓ Converts action → infusion via exponential scaling
#   ✓ Runs 20 seconds per action (you may increase)
#   ✓ Prints BIS, MAP, HR, Ce_prop, Ce_remi, Ce_nore
#
# After running:
#   You will see:
#       - BIS decrease with stronger propofol levels
#       - MAP decrease/increase depending on drug mix
#       - HR modulated by opioid stress + SU-model
#       - Ce concentrations rising realistically
#       - Disturbance-driven hemodynamic fluctuations
# --------------------------------------------------------------

from triagent_controller.pas_interface import PASInterface
from triagent_controller.state_action import action_to_infusions, NUM_ACTIONS

import time


def run_phase2_test():
    print("\n==========================================")
    print("       PHASE 2 — FULL ACTION SWEEP")
    print("   PAS + PK + PD + Disturbances ACTIVE")
    print("==========================================\n")

    env = PASInterface(dt=1.0)
    obs = env.reset()

    print("Initial State:")
    print(obs)
    print("\n------------------------------------------\n")

    # Loop all 27 discrete joint actions
    for action_id in range(NUM_ACTIONS):
        print(f"\n========== ACTION {action_id} of {NUM_ACTIONS-1} ==========")

        inf = action_to_infusions(action_id)
        print(f"Infusion rates (mg/s or µg/s): {inf}\n")

        # Run 20 seconds with this action
        for t in range(20):
            obs = env.step(inf)

            print(
                f"t={int(obs['time']):3d}s | "
                f"BIS={obs['BIS']:.1f} | "
                f"MAP={obs['MAP']:.1f} | "
                f"HR={obs['HR']:.1f} | "
                f"Ce_prop={obs['Ce_prop']:.3f} | "
                f"Ce_remi={obs['Ce_remi']:.3f} | "
                f"Ce_nore={obs['Ce_nore']:.3f}"
            )

        print("------------------------------------------")

    print("\n\n========== PHASE 2 TEST COMPLETE ==========\n")


if __name__ == "__main__":
    run_phase2_test()
