import bisect
import numpy as np
import cantera as ct
import Globals

class Value:
    """
    Base class for time-varying quantities (can hold any type).
    """

    def __init__(self, name: str, constant=None, times=None, values=None):
        self.name = name
        self._constant = True
        self._value = None
        self.times = []
        self.values = []

        if constant is not None:
            self.set(constant)
        elif times is not None and values is not None:
            self.set((times, values))

    def set(self, data):
        # Case 1: single constant (any type)
        if not isinstance(data, (tuple, list)):
            self._constant = True
            self._value = data
            self.times = [0.0]
            self.values = [data]
            return

        # Case 2: list of (t, v) pairs
        if isinstance(data, list) and all(isinstance(p, tuple) and len(p) == 2 for p in data):
            self._constant = False
            self.times = [float(t) for t, _ in data]
            self.values = [v for _, v in data]  # allow any type
            return

        # Case 3: tuple of arrays/lists
        if isinstance(data, tuple) and len(data) == 2:
            times, values = data
            if len(times) != len(values):
                raise ValueError("times and values must have the same length")
            self._constant = False
            self.times = list(map(float, times))
            self.values = list(values)  # allow any type
            return

        raise TypeError("Unsupported assignment")

    def at(self, t):
        """Return value at a specific time."""
        if self._constant:
            return self._value
        if not self.times:
            return None
        idx = bisect.bisect_right(self.times, t) - 1
        if idx < 0:
            return self.values[0]
        return self.values[idx]

    @property
    def value(self):
        """Value at the global simulation time."""
        return self.at(Globals.get_time())

    def __call__(self, new_value=None, t=None):
        """
        If called with no arguments, return the current value.
        If called with new_value, set it at the current global time (or supplied t).
        """
        if new_value is None:
            return self.value
        else:
            if t is None:
                t = Globals.get_time()
            self[t] = new_value   # uses __setitem__
            return new_value

    def __getitem__(self, t):
        return self.at(t)

    def __setitem__(self, t, val):
        if self._constant:
            # turn constant into time series
            self._constant = False
            self.times = [0.0]
            self.values = [self._value]
            self._value = None
        idx = bisect.bisect_left(self.times, t)
        if idx < len(self.times) and self.times[idx] == t:
            self.values[idx] = val
        else:
            self.times.insert(idx, float(t))
            self.values.insert(idx, val)

    @property
    def history(self):
        return [float(t) for t in self.times], list(self.values)

    def __repr__(self):
        if self._constant:
            return f"<{self.__class__.__name__} {self.name}: constant={self._value}>"
        return f"<{self.__class__.__name__} {self.name}: n_points={len(self.times)}>"
    
class State(Value):
    def set(self, data):
        if isinstance(data, (int, float, np.floating)):
            super().set(float(data))
        elif isinstance(data, (tuple, list)):
            # enforce numeric in time series
            if isinstance(data, list):  # [(t, v), ...]
                super().set([(t, float(v)) for t, v in data])
            else:  # (times, values)
                times, values = data
                super().set((times, [float(v) for v in values]))
        else:
            raise TypeError("State must be numeric")

class Parameter(Value):
    def set(self, data):
        if isinstance(data, (int, float, np.floating)):
            super().set(float(data))
        elif isinstance(data, (tuple, list)):
            if isinstance(data, list):  # [(t, v), ...]
                super().set([(t, float(v)) for t, v in data])
            else:  # (times, values)
                times, values = data
                super().set((times, [float(v) for v in values]))
        else:
            raise TypeError("Parameter must be numeric")


class Substance(Value):
    def set(self, data):
        def check(val):
            if not isinstance(val, ct._cantera.ThermoPhase):
                raise TypeError(f"Fluid must be a Cantera ThermoPhase, got {type(val)}")
            return val

        if isinstance(data, ct._cantera.ThermoPhase):
            super().set(check(data))
        elif isinstance(data, list):  # [(t, thermo), ...]
            super().set([(t, check(v)) for t, v in data])
        elif isinstance(data, tuple) and len(data) == 2:
            times, values = data
            super().set((times, [check(v) for v in values]))
        else:
            raise TypeError("Unsupported assignment for Fluid")


