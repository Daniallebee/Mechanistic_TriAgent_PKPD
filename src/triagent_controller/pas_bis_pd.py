# --------------------------------------------------------------
# pas_bis_pd.py — BIS PD model with Propofol–Remifentanil synergy
# --------------------------------------------------------------
# This file overrides the native PAS Eleveld BIS model
# by adding a second drug interaction (Remifentanil) using
# the exponential modulation model:
#
#       BIS = BIS_propofol * exp(-k_int * Ce_remi)
#
# This behaves smoothly, is RL-friendly, and clinically consistent.
# --------------------------------------------------------------

from __future__ import annotations
import numpy as np

# Import PAS-native components
from python_anesthesia_simulator.pd_models import EleveldBisPD


class SynergisticBisModel(EleveldBisPD):
    """
    Extends PAS Eleveld BIS PD model to include Remifentanil synergy.

    BIS = f_prop(Ce_prop) * exp(-k_int * Ce_remi)

    Parameters
    ----------
    patient : PAS Patient instance
    k_int : float
        Interaction coefficient controlling synergy intensity.
        Typical clinical range: 0.05–0.25
    """

    def __init__(self, patient, k_int: float = 0.12):
        super().__init__(patient)
        self.k_int = k_int
        self.Ce_remi = 0.0   # gets updated externally

    # ------------------------------------------------------------------
    # External PASInterface must update Ce_remi every step:
    #       bis_model.update_remi_effect(Ce_remi)
    # ------------------------------------------------------------------
    def update_remi_effect(self, Ce_remi: float):
        self.Ce_remi = float(max(Ce_remi, 0.0))

    # ------------------------------------------------------------------
    # Overriding the native BIS computation
    # ------------------------------------------------------------------
    def compute_effect(self, Ce_prop):
        """
        Computes synergistic BIS given Ce_prop (mg/L-ish)
        and previously set Ce_remi.
        """
        # Native Eleveld hypnotic depression curve
        bis_prop = super().compute_effect(Ce_prop)

        # Exponential synergy modulation
        synergy = np.exp(-self.k_int * self.Ce_remi)

        bis_out = bis_prop * synergy

        # Safety clamp
        bis_out = float(np.clip(bis_out, 5.0, 100.0))

        self.current_effect = bis_out
        return bis_out
