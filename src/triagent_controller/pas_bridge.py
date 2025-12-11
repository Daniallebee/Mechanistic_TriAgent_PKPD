# --------------------------------------------------------------
# pas_interface.py  —  UPDATED VERSION WITH BIS–REMI SYNERGY
# --------------------------------------------------------------
# This file now integrates:
#   - PK:   Eleveld (Propofol), Minto (Remi), Beloeil (Norepi)
#   - PD:   SynergisticBisModel (Prop + Remi synergy)
#           Ursino MAP
#           Ursino HR
#
# It uses your original PAS components and adds:
#      pd_bis.update_remi_effect()
#
# ABSOLUTELY NO CODE IS DELETED — only extended.
# --------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Dict, Any

# PAS imports (unchanged)
from python_anesthesia_simulator.patient import Patient
from python_anesthesia_simulator.simulator import Simulator
from python_anesthesia_simulator.disturbances import Disturbances
from python_anesthesia_simulator.pk_models import CompartmentModel
from python_anesthesia_simulator.pd_models import (
    UrsinoMapModel,
    UrsinoHRModel,
)

# New synergistic BIS PD model
from triagent_controller.pas_bis_pd import SynergisticBisModel


class PASInterface:
    """
    A clean interface for the PAS simulation engine.

    Provides:
        obs = reset()
        obs = step({"propofol": x, "remifentanil": y, "norepinephrine": z})

    Returns dict containing:
        BIS, MAP, HR
        Ce_prop, Ce_remi, Ce_nore
        time (seconds)
    """

    def __init__(
        self,
        dt: float = 1.0,
        disturbance_profile: str = "synthetic",
        obesity_factor: float = 1.0,
    ):
        self.dt = dt

        # ------------------------------------------------------
        # 1) Create synthetic patient (unchanged)
        # ------------------------------------------------------
        self.patient = Patient.synthetic_patient()
        self.patient.weight *= obesity_factor

        age = self.patient.age
        height = self.patient.height
        weight = self.patient.weight
        sex = 1 if self.patient.gender == "M" else 0
        lbm = self.patient.lean_body_mass

        patient_characteristics = [age, height, weight, sex]

        # ------------------------------------------------------
        # 2) Create PK models (unchanged)
        # ------------------------------------------------------
        self.pk_prop = CompartmentModel(
            Patient_characteristic=patient_characteristics,
            lbm=lbm,
            drug="Propofol",
            model="Eleveld",
            ts=dt,
        )

        self.pk_remi = CompartmentModel(
            Patient_characteristic=patient_characteristics,
            lbm=lbm,
            drug="Remifentanil",
            model="Minto",
            ts=dt,
        )

        self.pk_nore = CompartmentModel(
            Patient_characteristic=patient_characteristics,
            lbm=lbm,
            drug="Norepinephrine",
            model="Beloeil",
            ts=dt,
        )

        # ------------------------------------------------------
        # 3) PAS PD models (UPDATED BIS ONLY)
        # ------------------------------------------------------
        self.pd_bis = SynergisticBisModel(self.patient, k_int=0.12)
        self.pd_map = UrsinoMapModel(self.patient)
        self.pd_hr = UrsinoHRModel(self.patient)

        # ------------------------------------------------------
        # 4) Disturbances (unchanged)
        # ------------------------------------------------------
        self.dist = Disturbances(
            dist_profil=disturbance_profile,
            start_intub_time=0,
            start_surgery_time=60,
        )

        # ------------------------------------------------------
        # 5) PAS Simulator (unchanged)
        # ------------------------------------------------------
        self.sim = Simulator(
            patient=self.patient,
            pk_models=[self.pk_prop, self.pk_remi, self.pk_nore],
            pd_models=[self.pd_bis, self.pd_map, self.pd_hr],
            disturbances=self.dist,
            dt=dt,
        )

        self.time = 0.0

    # ----------------------------------------------------------
    # Reset simulation
    # ----------------------------------------------------------
    def reset(self) -> Dict[str, float]:
        self.sim.reset()
        self.time = 0.0
        return self.get_observation()

    # ----------------------------------------------------------
    # One simulation step
    # ----------------------------------------------------------
    def step(self, infusions: Dict[str, float]) -> Dict[str, float]:
        """
        infusions = {
            "propofol": x,
            "remifentanil": y,
            "norepinephrine": z
        }
        Units must match PAS expectations (mg/s, ug/s, etc.)
        """

        # ---- Infusion vector ----
        u = np.array([
            infusions["propofol"],
            infusions["remifentanil"],
            infusions["norepinephrine"],
        ])

        # ---- Step the PAS engine ----
        self.sim.step(u)
        self.time += self.dt

        # ---- Update BIS synergy with Remi effect-site ----
        Ce_remi = float(self.pk_remi.y[0, 0])
        self.pd_bis.update_remi_effect(Ce_remi)

        return self.get_observation()

    # ----------------------------------------------------------
    # Extract PAS → clean output
    # ----------------------------------------------------------
    def get_observation(self) -> Dict[str, float]:
        return {
            "BIS": float(self.pd_bis.current_effect),
            "MAP": float(self.pd_map.current_effect),
            "HR": float(self.pd_hr.current_effect),

            "Ce_prop": float(self.pk_prop.y[0, 0]),
            "Ce_remi": float(self.pk_remi.y[0, 0]),
            "Ce_nore": float(self.pk_nore.y[0, 0]),

            "time": self.time,
        }
