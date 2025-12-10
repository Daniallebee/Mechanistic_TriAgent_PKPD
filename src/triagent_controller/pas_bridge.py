from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .state_action import PatientState, JointAction, load_config


@dataclass
class PKPDParameters:
    """
    Tunable PK/PD parameters for the simplified tri-agent model.

    Units are deliberately "engineering friendly" rather than strictly
    pharmacological. What matters here is *qualitative behaviour*:

    - Propofol + Remifentanil push BIS down and MAP down
    - Norepinephrine pushes MAP up
    - All three have first-order washout (exponential decay)
    """
    # First-order elimination rates (per minute)
    ke_prop: float = 0.25
    ke_remi: float = 0.35
    ke_nore: float = 0.5

    # Input gains: how strongly infusion (action) feeds the effect-site
    gain_prop: float = 0.6
    gain_remi: float = 0.4
    gain_nore: float = 0.3

    # BIS model
    bis_min: float = 30.0     # deep anesthesia floor
    w_prop_bis: float = 1.0
    w_remi_bis: float = 0.7
    c50_bis: float = 1.5      # "C50" for combined hypnotic effect
    gamma_bis: float = 2.0    # Hill coefficient

    # MAP model
    w_prop_map: float = 0.7
    w_remi_map: float = 0.5
    w_nore_map: float = 1.2
    k_map_hyp: float = 12.0   # hypotensive strength of propo+remi
    k_map_press: float = 18.0 # pressor strength of nore

    # Heart rate baroreflex gain
    k_hr_baroreflex: float = 0.8  # bpm per mmHg


class PKPDModel:
    """
    Lightweight PK/PD engine for Phase 2.

    This class:

    * Reads the simulation time step and safety limits from problem.yaml
    * Evolves propofol / remifentanil / norepinephrine effect-site
      concentrations with simple 1-compartment kinetics
    * Maps effect-site concentrations to BIS and MAP
    * Applies a simple baroreflex on HR
    * Returns an updated PatientState + observable dict

    It **does not** talk directly to the PAS / Simulator code yet – this
    is an intentionally trimmed model suitable for RL training and for
    plugging into your later PAS-based Phase 3.
    """

    def __init__(self, params: PKPDParameters | None = None) -> None:
        cfg = load_config()
        # delta_t is specified in seconds in problem.yaml
        self.dt_sec: float = float(cfg["delta_t"])
        self.dt_min: float = self.dt_sec / 60.0
        self.map_floor: float = float(cfg.get("MAP_min", 65.0))

        self.params = params or PKPDParameters()

        # Baselines captured from the very first state we see
        self._initialized: bool = False
        self._bis_baseline: float = 95.0
        self._map_baseline: float = 80.0
        self._hr_baseline: float = 70.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_baseline(self, state: PatientState) -> None:
        if self._initialized:
            return
        # Take baselines from the initial state, with sane fallbacks
        self._bis_baseline = state.BIS if state.BIS > 0 else 95.0
        self._map_baseline = state.MAP if state.MAP > 0 else 80.0
        self._hr_baseline = state.HR if state.HR > 0 else 70.0
        self._initialized = True

    def _update_effect_site(
        self,
        Ce: float,
        u: float,
        ke_min: float,
        gain: float,
    ) -> float:
        """
        First-order compartment with affine input:

            dCe/dt = -ke * Ce + gain * u

        Integrated with forward Euler in minutes.
        """
        dt = self.dt_min
        return Ce + dt * (-ke_min * Ce + gain * u)

    def _compute_bis(self, Ce_prop: float, Ce_remi: float) -> float:
        p = self.params
        # Combined hypnotic / analgesic "drive"
        Ce_eff = p.w_prop_bis * Ce_prop + p.w_remi_bis * Ce_remi
        Ce_eff = max(Ce_eff, 0.0)
        # Hill-type saturation
        if Ce_eff <= 0.0:
            E = 0.0
        else:
            num = Ce_eff ** p.gamma_bis
            den = num + p.c50_bis ** p.gamma_bis
            E = num / den

        bis = self._bis_baseline - E * (self._bis_baseline - p.bis_min)
        # Clamp to plausible BIS range
        return float(min(100.0, max(0.0, bis)))

    def _compute_map(self, Ce_prop: float, Ce_remi: float, Ce_nore: float) -> float:
        p = self.params
        hypotension = p.k_map_hyp * (
            p.w_prop_map * Ce_prop + p.w_remi_map * Ce_remi
        )
        pressor = p.k_map_press * (p.w_nore_map * Ce_nore)

        map_val = self._map_baseline - hypotension + pressor
        # Apply safety floor from problem.yaml but allow a bit of undershoot
        lower_bound = self.map_floor - 15.0
        map_val = float(min(140.0, max(lower_bound, map_val)))
        return map_val

    def _compute_hr(self, map_val: float) -> float:
        p = self.params
        # Simple baroreflex: HR rises when MAP drops below baseline
        delta_map = self._map_baseline - map_val
        hr = self._hr_baseline + p.k_hr_baroreflex * delta_map
        return float(min(150.0, max(35.0, hr)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(
        self,
        state: PatientState,
        action: JointAction,
    ) -> Tuple[PatientState, Dict[str, float]]:
        """
        Advance the PK/PD state by one delta_t step.

        Parameters
        ----------
        state : PatientState
            The current patient state at time t.
        action : JointAction
            Discrete joint action: the three dose levels for
            (propofol, remifentanil, norepinephrine) chosen at this step.

        Returns
        -------
        new_state : PatientState
            Updated patient state at time t + delta_t.
        obs : dict
            Observable values (BIS, MAP, HR, Ce_prop, Ce_remi, Ce_nore).
        """
        self._ensure_baseline(state)

        p = self.params

        # ------------------------------------------------------------------
        # 1) PK: update effect-site concentrations for each drug
        # ------------------------------------------------------------------
        u_prop = float(action.propofol)
        u_remi = float(action.remifentanil)
        u_nore = float(action.norepinephrine)

        Ce_prop_new = self._update_effect_site(
            Ce=state.Ce_prop,
            u=u_prop,
            ke_min=p.ke_prop,
            gain=p.gain_prop,
        )
        Ce_remi_new = self._update_effect_site(
            Ce=state.Ce_remi,
            u=u_remi,
            ke_min=p.ke_remi,
            gain=p.gain_remi,
        )
        Ce_nore_new = self._update_effect_site(
            Ce=state.Ce_nore,
            u=u_nore,
            ke_min=p.ke_nore,
            gain=p.gain_nore,
        )

        # Avoid negative concentrations from numerical issues
        Ce_prop_new = max(Ce_prop_new, 0.0)
        Ce_remi_new = max(Ce_remi_new, 0.0)
        Ce_nore_new = max(Ce_nore_new, 0.0)

        # ------------------------------------------------------------------
        # 2) PD: map concentrations to BIS, MAP, HR
        # ------------------------------------------------------------------
        bis_new = self._compute_bis(Ce_prop_new, Ce_remi_new)
        map_new = self._compute_map(Ce_prop_new, Ce_remi_new, Ce_nore_new)
        hr_new = self._compute_hr(map_new)

        # For Phase 2 we keep stimulus and temperature fixed; later phases
        # (surgical stimulation model, thermoregulation) can update them.
        stimulus_new = state.stimulus
        temp_new = state.temperature

        # ------------------------------------------------------------------
        # 3) Build next PatientState and observation dict
        # ------------------------------------------------------------------
        new_state = PatientState(
            # Demographics stay fixed for a given episode
            age=state.age,
            weight=state.weight,
            height=state.height,
            gender=state.gender,
            # Updated physiology
            BIS=bis_new,
            MAP=map_new,
            HR=hr_new,
            Ce_prop=Ce_prop_new,
            Ce_remi=Ce_remi_new,
            Ce_nore=Ce_nore_new,
            # Context
            stimulus=stimulus_new,
            temperature=temp_new,
        )

        obs = {
            "BIS": bis_new,
            "MAP": map_new,
            "HR": hr_new,
            "Ce_prop": Ce_prop_new,
            "Ce_remi": Ce_remi_new,
            "Ce_nore": Ce_nore_new,
        }

        return new_state, obs
