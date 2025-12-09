"""
triagent_controller

Project-specific code for the mechanistic tri-agent (propofol, remifentanil,
norepinephrine) closed-loop control system built on top of the
Python Anesthesia Simulator (PAS).

Phase 1: this package defines the problem configuration and the
state/action structures. Later phases will add controllers,
cost functions, and simulation pipelines.
"""
"""
triagent_controller

Mechanistic tri-agent closed-loop control system with RL support.
"""
from .state_action import PatientState, JointAction, enumerate_actions  # noqa: F401
from .rl_environment import AnesthesiaEnv  # Add this line

__all__ = ['PatientState', 'JointAction', 'enumerate_actions', 'AnesthesiaEnv']

