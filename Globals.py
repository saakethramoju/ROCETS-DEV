"""
Centralized global state for simulation.
Keeps track of the simulation time.
"""

# Internal storage
_time = 0.0

def set_time(t: float):
    """Update the global simulation time."""
    global _time
    _time = float(t)

def get_time() -> float:
    """Retrieve the current simulation time."""
    return _time

def reset_time():
    global _time
    _time = 0.0

