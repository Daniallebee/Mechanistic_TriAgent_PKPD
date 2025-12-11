# --------------------------------------------------------------
# pas_interface.py — FINAL PAS v1 COMPATIBLE VERSION
# --------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Dict

from python_anesthesia_simulator.patient import Patient
from python_anesthesia_simulator.simulator import Simulator
from python_anesthesia_simulator.disturbances import Disturbances


class PASInterface:

    def __init__(
        self,
        dt: float = 1.0,
        disturbance_profile: str = "realistic",
        obesity_factor: float = 1.0,
    ):
        self.dt = dt
        self.disturbance_profile = disturbance_profile

        # -----------------------------
        # FIXED PATIENT
        # -----------------------------
        age = 35
        height = 170
        weight = 70 * obesity_factor
        sex = 0  # female

        patient_characteristics = [age, height, weight, sex]

        self.patient = Patient(
            patient_characteristic=patient_characteristics,
            ts=dt,
            model_propo="Eleveld",
            model_remi="Minto",
            model_nore="Beloeil",
            model_hemo="Su",
            model_bis="Bouillon"
        )

        # Attach disturbances
        self.patient.disturbances = Disturbances(
            dist_profil=disturbance_profile,
            start_intub_time=0,
            start_surgery_time=60,
        )

        # Simulator is required by PAS v1 but not used directly
        self.sim = Simulator(self.patient, dt)

        self.time = 0.0

    # ----------------------------------------------------------
    # RESET
    # ----------------------------------------------------------
    def reset(self) -> Dict[str, float]:

        age = self.patient.age
        height = self.patient.height
        weight = self.patient.weight
        sex = self.patient.sex

        patient_characteristics = [age, height, weight, sex]

        self.patient = Patient(
            patient_characteristic=patient_characteristics,
            ts=self.dt,
            model_propo="Eleveld",
            model_remi="Minto",
            model_nore="Beloeil",
            model_hemo="Su",
            model_bis="Bouillon"
        )

        # Reassign disturbances
        self.patient.disturbances = Disturbances(
            dist_profil=self.disturbance_profile,
            start_intub_time=0,
            start_surgery_time=60,
        )

        self.sim = Simulator(self.patient, self.dt)

        self.time = 0.0

        return self.get_observation()

    # ----------------------------------------------------------
    # STEP
    # ----------------------------------------------------------
    def step(self, infusions: Dict[str, float]) -> Dict[str, float]:

        u_propo = infusions["propofol"]
        u_remi = infusions["remifentanil"]
        u_nore = infusions["norepinephrine"]

        # -------------------------
        # PK updates
        # -------------------------
        self.patient.c_es_propo = self.patient.propo_pk.one_step(u_propo)
        self.patient.c_es_remi  = self.patient.remi_pk.one_step(u_remi)
        self.patient.c_blood_nore = self.patient.nore_pk.one_step(u_nore)

        # -------------------------
        # PD updates — BIS
        # -------------------------
        self.patient.bis = self.patient.bis_pd.one_step(
            self.patient.c_es_propo,
            self.patient.c_es_remi
        )

        # -------------------------
        # PD updates — HEMO
        # -------------------------
        hemo = self.patient.hemo_pd.one_step(
            cp_propo=self.patient.propo_pk.x[0, 0],
            cp_remi=self.patient.remi_pk.x[0, 0],
            cp_nore=self.patient.c_blood_nore,
            v_ratio=1.0,
            disturbances=[0, 0, 0]
        )

        self.patient.tpr = hemo[0]
        self.patient.sv  = hemo[1]
        self.patient.hr  = hemo[2]
        self.patient.map = hemo[3]
        self.patient.co  = hemo[4]

        self.time += self.dt

        return self.get_observation()

    # ----------------------------------------------------------
    # OBSERVATION
    # ----------------------------------------------------------
    def get_observation(self) -> Dict[str, float]:

        hemo = self.patient.hemo_pd.output_function(self.patient.hemo_pd.x_effect)

        if len(self.patient.bis_pd.bis_buffer) == 0:
            bis_val = float(self.patient.bis)
        else:
            bis_val = float(self.patient.bis_pd.bis_buffer[-1])

        return {
            "BIS": bis_val,
            "MAP": float(hemo[3]),
            "HR": float(hemo[2]),

            "Ce_prop": float(self.patient.propo_pk.x[0]),
            "Ce_remi": float(self.patient.remi_pk.x[0]),
            "Ce_nore": float(self.patient.nore_pk.x[0]),

            "Cp_prop": float(self.patient.propo_pk.x[-1]),
            "Cp_remi": float(self.patient.remi_pk.x[-1]),
            "Cp_nore": float(self.patient.nore_pk.x[-1]),

            "time": self.time,
        }
