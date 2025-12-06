"""
triagent_controller

Project-specific code for the mechanistic tri-agent (propofol, remifentanil,
norepinephrine) closed-loop control system built on top of the
Python Anesthesia Simulator (PAS).

Phase 1: this package defines the problem configuration and the
state/action structures. Later phases will add controllers,
cost functions, and simulation pipelines.
"""

from .state_action import PatientState, JointAction, enumerate_actions  # noqa: F401
