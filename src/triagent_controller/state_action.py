from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------

# Cached configuration dictionary loaded from problem.yaml
_CONFIG: Dict[str, Any] | None = None


def _config_path() -> Path:
    """
    Return the absolute path to the problem.yaml file that defines
    the Phase 1 problem parameters (time grid, targets, dose levels).
    """
    # problem.yaml lives in the same directory as this file
    return Path(__file__).with_name("problem.yaml")


def load_config() -> Dict[str, Any]:
    """
    Load and cache the Phase 1 configuration from problem.yaml.

    Returns
    -------
    dict
        Dictionary containing keys:
        - 'delta_t'
        - 'horizon'
        - 'duration_seconds'
        - 'target_BIS'
        - 'MAP_min'
        - 'dose_levels' (with 'propofol', 'remifentanil', 'norepinephrine')
    """
    global _CONFIG
    if _CONFIG is None:
        path = _config_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {path}. "
                "Expected problem.yaml to exist in triagent_controller/"
            )
        with path.open("r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def _dose_levels() -> Dict[str, List[float]]:
    """
    Return the discrete dose levels for each drug as lists of floats.

    Ensures that each drug has exactly three levels (low, medium, high),
    as required for the 3 x 3 x 3 = 27 action enumeration.
    """
    cfg = load_config()
    try:
        levels = cfg["dose_levels"]
        prop = list(levels["propofol"])
        remi = list(levels["remifentanil"])
        nore = list(levels["norepinephrine"])
    except KeyError as exc:
        raise KeyError(
            "Missing 'dose_levels' or one of "
            "['propofol', 'remifentanil', 'norepinephrine'] "
            "in problem.yaml"
        ) from exc

    if not (len(prop) == len(remi) == len(nore) == 3):
        raise ValueError(
            "Each of 'propofol', 'remifentanil', and 'norepinephrine' "
            "must have exactly 3 dose levels in problem.yaml."
        )

    return {
        "propofol": [float(x) for x in prop],
        "remifentanil": [float(x) for x in remi],
        "norepinephrine": [float(x) for x in nore],
    }


# ---------------------------------------------------------------------------
# Patient state definition
# ---------------------------------------------------------------------------


@dataclass
class PatientState:
    """
    Container for the patient state at a single time step.

    Phase 1: this is purely a data structure, with no PK/PD update logic.
    Later phases (PK/PD integration, cardiovascular model, stimulus
    generation) will use and update these fields.

    Demographics
    -----------
    age : float
        Age in years.
    weight : float
        Weight in kilograms.
    height : float
        Height in centimeters.
    gender : str
        Gender identifier (e.g. 'M', 'F', or other agreed convention).

    Physiology
    ----------
    BIS : float
        Bispectral Index (target range: 40–60).
    MAP : float
        Mean Arterial Pressure in mmHg (safety floor: ≥ 65).
    HR : float
        Heart rate in beats per minute.

    Ce_prop : float
        Effect-site concentration of propofol.
    Ce_remi : float
        Effect-site concentration of remifentanil.
    Ce_nore : float
        Effect-site concentration of norepinephrine.

    Surgical context
    ----------------
    stimulus : float
        Surgical stimulation level (e.g. 0–100 scale).
    temperature : float
        Core body temperature in degrees Celsius.
    """

    # Demographics
    age: float
    weight: float
    height: float
    gender: str

    # Physiology
    BIS: float
    MAP: float
    HR: float

    Ce_prop: float
    Ce_remi: float
    Ce_nore: float

    # Surgical context
    stimulus: float
    temperature: float


# ---------------------------------------------------------------------------
# Joint action definition and enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointAction:
    """
    Representation of a single joint action for the tri-agent controller.

    Each action consists of a discrete choice of dose level for:
      - propofol
      - remifentanil
      - norepinephrine

    There are 3 levels per drug (low, medium, high), giving
    3 x 3 x 3 = 27 joint actions in total.

    The 'id' field uses the indexing convention:

        id = 9 * k + 3 * l + m

    where k, l, m are the (0-based) indices of the dose level for
    propofol, remifentanil, and norepinephrine respectively.
    """

    id: int

    # 0-based indices of the dose levels
    k: int  # propofol index
    l: int  # remifentanil index
    m: int  # norepinephrine index

    # Actual dose values (units as per problem.yaml)
    propofol: float
    remifentanil: float
    norepinephrine: float


def enumerate_actions() -> List[JointAction]:
    """
    Enumerate all 27 joint actions for the tri-agent controller.

    Returns
    -------
    List[JointAction]
        A list of 27 JointAction objects, ordered according to the
        indexing rule:

            id = 9 * k + 3 * l + m

        with k, l, m in {0, 1, 2}.

    Notes
    -----
    - No drug is fixed: all three drugs vary jointly across their
      three discrete levels.
    - This function performs internal sanity checks to ensure that:
        * exactly 27 unique (propofol, remifentanil, norepinephrine)
          triples are generated.
    """
    levels = _dose_levels()
    prop_levels = levels["propofol"]
    remi_levels = levels["remifentanil"]
    nore_levels = levels["norepinephrine"]

    actions: List[JointAction] = []

    for k in range(3):  # propofol index
        for l in range(3):  # remifentanil index
            for m in range(3):  # norepinephrine index
                action_id = 9 * k + 3 * l + m
                actions.append(
                    JointAction(
                        id=action_id,
                        k=k,
                        l=l,
                        m=m,
                        propofol=prop_levels[k],
                        remifentanil=remi_levels[l],
                        norepinephrine=nore_levels[m],
                    )
                )

    # Sanity checks
    if len(actions) != 27:
        raise RuntimeError(
            f"Expected 27 actions, got {len(actions)}. "
            "Check the dose levels definition."
        )

    unique_triples = {
        (a.propofol, a.remifentanil, a.norepinephrine) for a in actions
    }
    if len(unique_triples) != 27:
        raise RuntimeError(
            "Dose level enumeration produced non-unique joint actions. "
            "Check the dose levels in problem.yaml."
        )

    return actions
