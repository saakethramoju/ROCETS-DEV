import Globals

class Balance:
    """
    Represents an algebraic balance constraint:
        residual(independent, dep1, dep2) = 0
    with time-dependent variables and activation intervals.

    Example:
        Balance(Cd, mdot, 5.0)  
            → residual = mdot - 5

        Balance(Cd, mdot, 5.0, 
                residual_fn=lambda indep, x, y: indep * x - y)
    """

    def __init__(self, independent, dep1, dep2, 
                 active_intervals=None, residual_fn=None):
        self.independent = independent  # Value (time-dependent)
        self.dep1 = dep1                # Value (time-dependent)
        self.dep2 = dep2                # Value (time-dependent or constant)

        # [(t_start, t_end), ...] ; if None, always active
        if active_intervals is None:
            self.active_intervals = [(-float("inf"), float("inf"))]
        else:
            self.active_intervals = active_intervals

        # Default residual: dep1 - dep2
        self.residual_fn = residual_fn or (lambda indep, x, y: x - y)

    def is_active(self, t=None):
        """Check if the balance is active at time t (default = current sim time)."""
        if t is None:
            t = Globals.get_time()
        return any(start <= t <= end for start, end in self.active_intervals)

    def residual(self, t=None):
        """Return the residual at time t, if active, else 0."""
        if t is None:
            t = Globals.get_time()

        if not self.is_active(t):
            return 0.0

        indep_val = self.independent[t]
        dep1_val = self.dep1[t]
        dep2_val = self.dep2[t] if hasattr(self.dep2, "__getitem__") else self.dep2

        return self.residual_fn(indep_val, dep1_val, dep2_val)

    def __str__(self):
        return (f"Balance(independent={self.independent.name}, "
                f"dep1={getattr(self.dep1, 'name', str(self.dep1))}, "
                f"dep2={getattr(self.dep2, 'name', str(self.dep2))}, "
                f"active_intervals={self.active_intervals}, "
                f"residual_fn={self.residual_fn.__name__ if hasattr(self.residual_fn, '__name__') else 'custom'})")

    def __repr__(self):
        return (f"<Balance: {self.independent.name} | "
                f"{getattr(self.dep1, 'name', str(self.dep1))} vs "
                f"{getattr(self.dep2, 'name', str(self.dep2))}>")


    def set_independent(self, value, t=None):
        """Set the independent variable to a new value (for solver trial steps)."""
        if t is None:
            t = Globals.get_time()
        self.independent[t] = value