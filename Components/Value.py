import bisect
import numpy as np
import Globals   # root-level Globals.py

class Value:
    """
    Base class for time-varying quantities (State, Parameter).
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
        # Case 1: scalar constant
        if isinstance(data, (int, float, np.floating)):
            self._constant = True
            self._value = float(data)
            self.times = [0.0]
            self.values = [self._value]
            return

        # Case 2: list of (t, v) pairs
        if isinstance(data, list) and all(isinstance(p, tuple) and len(p) == 2 for p in data):
            self._constant = False
            self.times = [float(t) for t, _ in data]
            self.values = [float(v) for _, v in data]
            return

        # Case 3: tuple of arrays/lists
        if isinstance(data, tuple) and len(data) == 2:
            times, values = data
            if len(times) != len(values):
                raise ValueError("times and values must have the same length")
            self._constant = False
            self.times = list(map(float, times))
            self.values = list(map(float, values))
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

    def __call__(self):
        """Alias for .value, so you can do obj()."""
        return self.value

    def __getitem__(self, t):
        return self.at(t)

    def __setitem__(self, t, val):
        if self._constant:
            self._constant = False
            self.times = [0.0]
            self.values = [self._value]
            self._value = None
        idx = bisect.bisect_left(self.times, t)
        if idx < len(self.times) and self.times[idx] == t:
            self.values[idx] = float(val)
        else:
            self.times.insert(idx, float(t))
            self.values.insert(idx, float(val))

    @property
    def history(self):
        return [float(t) for t in self.times], [float(v) for v in self.values]

    def __repr__(self):
        if self._constant:
            return f"<{self.__class__.__name__} {self.name}: constant={self._value}>"
        return f"<{self.__class__.__name__} {self.name}: n_points={len(self.times)}>"


class State(Value):
    pass

class Parameter(Value):
    pass

