"""
Final calibrated RL environment with correct drug dosing for the tri-agent anesthesia controller.
"""

import numpy as np
import gym
from gym import spaces
from typing import Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd

from python_anesthesia_simulator import Patient, Simulator
from .state_action import load_config

class AnesthesiaEnv(gym.Env):
    """OpenAI Gym environment for anesthesia control - Final Calibration"""
    
    def __init__(self, 
                 use_discrete_actions: bool = False,
                 patient_distribution: str = 'uniform',
                 episode_length_minutes: int = 60,
                 verbose: bool = False):
        super().__init__()
        
        # Load configuration
        self.config = load_config()
        self.ts = self.config['delta_t']  # 5 seconds
        self.episode_length = episode_length_minutes * 60 // self.ts
        self.verbose = verbose
        
        # Define action space (continuous for 3 drugs)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(10,), dtype=np.float32
        )
        
        # Patient distribution
        self.patient_distribution = patient_distribution
        
        # Initialize tracking
        self.simulator = None
        self.current_step = 0
        self.episode_data = []
        
    def reset(self, patient_params: Optional[list] = None) -> np.ndarray:
        """Reset environment with new patient"""
        
        # Create patient
        if patient_params:
            patient = Patient(patient_params, ts=self.ts)
        else:
            # Default patient for testing
            patient = Patient([35, 170, 70, 1], ts=self.ts)  # 35yr, 170cm, 70kg, male
        
        self.simulator = Simulator(
            patient, 
            disturbance_profil=None,  # Start without disturbances for testing
            noise=False                # No noise initially for testing
        )
        
        self.current_step = 0
        self.episode_data = []
        
        if self.verbose:
            print(f"Reset: Patient age={patient.age}, weight={patient.weight}kg, BIS={patient.bis:.1f}")
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return transition"""
        
        # Clip actions to valid range
        action = np.clip(action, -1, 1)
        
        # Get patient weight
        weight = self.simulator.patient.weight
        
        # CAREFULLY CALIBRATED DRUG RATES
        # Based on the test showing 2 mg/s causes BIS to drop to 15 in 20 steps (100s)
        # We need much lower rates for realistic control
        
        # Propofol: 0-0.5 mg/s (reduced from 0-4)
        # This gives us finer control and prevents overshooting
        u_propo = self._scale_action(action[0], 0, 0.5)
        
        # Remifentanil: 0-0.3 µg/s (reduced from 0-1.5)
        u_remi = self._scale_action(action[1], 0, 0.3)
        
        # Norepinephrine: 0-0.1 µg/s (reduced from 0-0.3)
        u_nore = self._scale_action(action[2], 0, 0.1)
        
        # Simulate one step
        bis, map_val, hr, tof = self.simulator.one_step(
            input_propo=u_propo,
            input_remi=u_remi,
            input_nore=u_nore
        )
        
        # Store data for analysis
        self.episode_data.append({
            'step': self.current_step,
            'time_min': self.current_step * self.ts / 60,
            'bis': bis,
            'map': map_val,
            'hr': hr,
            'u_propo': u_propo,
            'u_remi': u_remi,
            'u_nore': u_nore,
            'action_0': action[0],
            'action_1': action[1],
            'action_2': action[2]
        })
        
        # Calculate reward
        reward = self._calculate_reward(bis, map_val, hr)
        
        # Check termination
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # Safety termination
        if map_val < 50:  # Severe hypotension
            done = True
            reward -= 10  # Large penalty
            if self.verbose:
                print(f"Episode terminated: MAP too low ({map_val:.1f})")
        
        if bis < 5:  # Extremely deep anesthesia
            reward -= 10  # Large penalty for dangerous depth
            if self.verbose:
                print(f"Warning: BIS critically low ({bis:.1f})")
        
        info = {
            'bis': bis,
            'map': map_val,
            'hr': hr,
            'drug_rates': [u_propo, u_remi, u_nore]
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Extract current state"""
        
        if self.simulator is None:
            return np.zeros(10, dtype=np.float32)
            
        patient = self.simulator.patient
        
        if len(self.simulator.dataframe) == 0:
            # Initial state
            return np.array([
                patient.bis / 100,
                patient.map / 100,
                patient.hr / 100,
                patient.co / 10,
                0, 0, 0,  # No drugs yet
                0,        # TOL
                1,        # TOF (100%)
                0         # Time = 0
            ], dtype=np.float32)
        
        df = self.simulator.dataframe
        obs = np.array([
            df['BIS'].iloc[-1] / 100,
            df['MAP'].iloc[-1] / 100,
            df['HR'].iloc[-1] / 100,
            df['CO'].iloc[-1] / 10,
            df['x_propo_4'].iloc[-1] / 10,
            df['x_remi_4'].iloc[-1] / 10,
            df['x_nore_1'].iloc[-1] / 10,
            df['TOL'].iloc[-1],
            df['TOF'].iloc[-1] / 100,
            self.current_step / self.episode_length
        ], dtype=np.float32)
        
        return obs
    
    def _scale_action(self, action: float, min_val: float, max_val: float) -> float:
        """Scale action from [-1, 1] to [min_val, max_val]"""
        return min_val + (action + 1) * (max_val - min_val) / 2
    
    def _calculate_reward(self, bis: float, map_val: float, hr: float) -> float:
        """Calculate reward based on clinical targets"""
        
        reward = 0.0
        
        # BIS target (40-60) - main control objective
        if 40 <= bis <= 60:
            reward += 2.0  # High reward for being in target
        elif 35 <= bis < 40 or 60 < bis <= 65:
            reward += 1.0  # Acceptable range
        elif 30 <= bis < 35 or 65 < bis <= 70:
            reward += 0.2
        elif 25 <= bis < 30 or 70 < bis <= 75:
            reward -= 0.5
        elif 20 <= bis < 25 or 75 < bis <= 80:
            reward -= 1.0
        elif 10 <= bis < 20 or 80 < bis <= 90:
            reward -= 2.0
        else:  # bis < 10 or bis > 90
            reward -= 3.0
        
        # MAP safety (>65) - critical safety constraint
        if map_val >= 75:
            reward += 1.0  # Optimal MAP
        elif 65 <= map_val < 75:
            reward += 0.5  # Safe MAP
        elif 60 <= map_val < 65:
            reward += 0.0  # Borderline
        else:
            reward -= (65 - map_val) / 5  # Increasing penalty below safety threshold
        
        # HR normal range (50-90)
        if 60 <= hr <= 80:
            reward += 0.3  # Optimal HR
        elif 50 <= hr < 60 or 80 < hr <= 90:
            reward += 0.1
        elif 45 <= hr < 50 or 90 < hr <= 100:
            reward += 0.0
        else:
            reward -= 0.5
        
        return reward
    
    def get_episode_data(self) -> pd.DataFrame:
        """Get episode data as DataFrame for analysis"""
        return pd.DataFrame(self.episode_data) if self.episode_data else pd.DataFrame()