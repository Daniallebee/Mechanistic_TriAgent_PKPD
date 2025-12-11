# --------------------------------------------------------------
# state_action.py  —  FINAL VERSION (S2 exponential scaling)
# --------------------------------------------------------------

import numpy as np


# --------------------------------------------------------------
# EXPONENTIAL SCALING FUNCTION — Option S2
# --------------------------------------------------------------
def exp_scale(level: int, max_rate: float, k: float = 1.5) -> float:
    """
    Maps discrete integer levels (0–3) to smooth infusion rates
    using exponential scaling.

        level = 0 → 0
        level = 1 → mild
        level = 2 → moderate
        level = 3 → strong

    Parameters
    ----------
    level : int
        Discrete dose level (0–3)
    max_rate : float
        Maximum allowed infusion rate for that drug
    k : float
        Exponential aggressiveness factor

    Returns
    -------
    float
        infusion rate in mg/s or ug/s (PAS units)
    """
    if level <= 0:
        return 0.0
    return max_rate * (1 - np.exp(-k * level)) / (1 - np.exp(-k * 3))


# --------------------------------------------------------------
# DRUG LIMITS (mg/s or µg/s)
# These values define the top limits for exponential scaling.
# They were chosen based on safe anesthetic infusion ranges.
# --------------------------------------------------------------

MAX_PROP_RATE = 1.2 / 60      # mg/s   (~72 mg/min peak)
MAX_REMI_RATE = 0.4           # µg/s   (~24 µg/min)
MAX_NORE_RATE = 0.20          # µg/s   (moderate vasopressor)


# --------------------------------------------------------------
# Enumerate the 27 possible actions
# Action triplet format: (prop_level, remi_level, nore_level)
# Each level is an integer 0–2 (3 options → 3^3 = 27 actions)
# --------------------------------------------------------------

def enumerate_actions():
    actions = []
    for p in range(3):        # propofol levels 0–2
        for r in range(3):    # remifentanil levels 0–2
            for n in range(3):  # norepinephrine levels 0–2
                actions.append((p, r, n))
    return actions


ACTIONS = enumerate_actions()     # list of 27 actions
NUM_ACTIONS = len(ACTIONS)        # = 27


# --------------------------------------------------------------
# Convert a discrete action ID → infusion dict (PAS-compatible)
# --------------------------------------------------------------

def action_to_infusions(action_id: int) -> dict:
    """
    Convert an integer action ID (0–26) into the real infusion
    rates for Propofol, Remifentanil, Norepinephrine.

    Output units match PAS expectations:
        Propofol      → mg/s   (Eleveld PK)
        Remifentanil  → µg/s   (Minto PK)
        Norepinephrine→ µg/s   (Beloeil PK)
    """

    p_level, r_level, n_level = ACTIONS[action_id]

    prop_rate = exp_scale(p_level, MAX_PROP_RATE)
    remi_rate = exp_scale(r_level, MAX_REMI_RATE)
    nore_rate = exp_scale(n_level, MAX_NORE_RATE)

    return {
        "propofol": float(prop_rate),
        "remifentanil": float(remi_rate),
        "norepinephrine": float(nore_rate),
    }
