"""
Quick Phase 2 smoke test.

This script:
    - Builds an initial PatientState
    - Instantiates the simplified PKPDModel
    - Enumerates the 27 joint actions
    - Applies one fixed action for a few steps
    - Prints BIS / MAP / HR and Ce_* so you can see dynamics

Run from the repo root (venv activated):

    python phase2_test.py
"""

from triagent_controller.pas_bridge import PKPDModel
from triagent_controller.state_action import (
    PatientState,
    enumerate_actions,
    load_config,
)


def build_initial_state() -> PatientState:
    cfg = load_config()
    # We only really need delta_t for later phases, but we read it here
    # to confirm problem.yaml is wired correctly.
    _ = cfg["delta_t"]

    # Reasonable "pre-induction" baselines
    return PatientState(
        age=40.0,
        weight=70.0,
        height=170.0,
        gender="F",      # convention: 'F' or 'M'
        BIS=95.0,
        MAP=80.0,
        HR=70.0,
        Ce_prop=0.0,
        Ce_remi=0.0,
        Ce_nore=0.0,
        stimulus=20.0,
        temperature=36.5,
    )


def main() -> None:
    state = build_initial_state()
    model = PKPDModel()

    actions = enumerate_actions()
    print("Total actions:", len(actions))

    # For this quick test, just use the *middle* joint action
    # (k=l=m=1) which should correspond to moderate doses.
    mid_action = actions[13]  # id = 9*1 + 3*1 + 1 = 13

    print("Using joint action id:", mid_action.id)
    print(
        f"  propofol={mid_action.propofol}, "
        f"remifentanil={mid_action.remifentanil}, "
        f"norepinephrine={mid_action.norepinephrine}"
    )
    print()

    num_steps = 10
    for t in range(num_steps):
        state, obs = model.step(state, mid_action)
        print(
            f"Step {t+1:02d} | "
            f"BIS={obs['BIS']:.1f}  "
            f"MAP={obs['MAP']:.1f}  "
            f"HR={obs['HR']:.1f}  "
            f"Ce_prop={obs['Ce_prop']:.3f}  "
            f"Ce_remi={obs['Ce_remi']:.3f}  "
            f"Ce_nore={obs['Ce_nore']:.3f}"
        )


if __name__ == "__main__":
    main()
