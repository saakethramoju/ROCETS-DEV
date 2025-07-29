from CoolProp.CoolProp import PropsSI
import CoolProp


class Fluid:
    _phase_name_map = {
        CoolProp.iphase_liquid: "Liquid",
        CoolProp.iphase_gas: "Gas",
        CoolProp.iphase_twophase: "Two-Phase",
        CoolProp.iphase_supercritical: "Supercritical",
        CoolProp.iphase_supercritical_gas: "Supercritical Gas",
        CoolProp.iphase_supercritical_liquid: "Supercritical Liquid",
        CoolProp.iphase_critical_point: "Critical Point",
        CoolProp.iphase_unknown: "Unknown",
        CoolProp.iphase_not_imposed: "Not Imposed"
    }

    def __init__(self, name: str = 'Water', *, T=None, P=None, X=None):
        self.name = name
        self._set_inputs(T, P, X)
        self._update_state()

    @property
    def is_mixture(self) -> bool:
        return False

    def _set_inputs(self, T, P, X):
        if (T is not None) + (P is not None) + (X is not None) != 2:
            raise ValueError("Exactly two of T, P, X must be provided.")

        if T is not None: self._T = T
        if P is not None: self._P = P
        if X is not None: self._X = X

        if T is not None and P is not None:
            self._input_pair = ("T", "P")
        elif T is not None and X is not None:
            self._input_pair = ("T", "Q")
        elif P is not None and X is not None:
            self._input_pair = ("P", "Q")

    def _update_state(self):
        if self._input_pair == ("T", "P"):
            self._state = {"T": self._T, "P": self._P}
        elif self._input_pair == ("T", "Q"):
            self._P = PropsSI("P", "T", self._T, "Q", self._X, self.name)
            self._state = {"T": self._T, "Q": self._X}
        elif self._input_pair == ("P", "Q"):
            self._T = PropsSI("T", "P", self._P, "Q", self._X, self.name)
            self._state = {"P": self._P, "Q": self._X}

    def _check_input_key(self, key):
        if key not in self._input_pair:
            raise AttributeError(f"Cannot set '{key}'. Only {self._input_pair} are allowed.")

    def _prop(self, key):
        input_map = {"T": "_T", "P": "_P", "Q": "_X"}
        k1, k2 = self._input_pair
        v1 = getattr(self, input_map[k1])
        v2 = getattr(self, input_map[k2])
        try:
            return PropsSI(key, k1, v1, k2, v2, self.name)
        except Exception:
            return float("nan")

    # --- Input Properties ---
    @property
    def T(self): return getattr(self, "_T", None)
    @T.setter
    def T(self, value):
        self._check_input_key("T")
        self._T = value
        self._update_state()

    @property
    def P(self): return getattr(self, "_P", None)
    @P.setter
    def P(self, value):
        self._check_input_key("P")
        self._P = value
        self._update_state()

    @property
    def X(self): return getattr(self, "_X", None)
    @X.setter
    def X(self, value):
        self._check_input_key("Q")
        self._X = value
        self._update_state()

    # --- Thermodynamic Properties ---
    @property
    def density(self): return self._prop("D")

    @property
    def enthalpy(self): return self._prop("H")

    @property
    def viscosity(self): return self._prop("V")

    @property
    def cp(self):
        value = self._prop("C")
        if value is None or value <= 0 or value != value:  # check for NaN using value != value
            return float("nan")
        return value

    @property
    def thermal_conductivity(self): return self._prop("L")

    @property
    def prandtl(self):
        value = self._prop("PRANDTL")
        if value is None or value <= 0 or value != value:
            return float("nan")
        return value

    @property
    def speed_of_sound(self): return self._prop("A")

    @property
    def phase(self):
        input_map = {"T": "_T", "P": "_P", "Q": "_X"}
        k1, k2 = self._input_pair
        v1 = getattr(self, input_map[k1])
        v2 = getattr(self, input_map[k2])
        try:
            # CoolProp returns float (e.g., 6.0), cast to int for mapping
            phase_int = int(PropsSI("Phase", k1, v1, k2, v2, self.name))
            return self._phase_name_map.get(phase_int, f"Unknown ({phase_int})")
        except Exception as e:
            return "Unknown"


    @property
    def molecular_weight(self): return PropsSI("M", "", 0, "", 0, self.name)

    # --- Saturation and Critical Properties ---
    @property
    def saturation_pressure(self): return PropsSI("P", "T", self._T, "Q", 0, self.name)

    @property
    def saturation_temperature(self): return PropsSI("T", "P", self._P, "Q", 0, self.name)

    @property
    def critical_temperature(self): return PropsSI("TCRIT", "", 0, "", 0, self.name)

    @property
    def critical_pressure(self): return PropsSI("PCRIT", "", 0, "", 0, self.name)

    @property
    def min_temperature(self): return PropsSI("TMIN", "", 0, "", 0, self.name)

    @property
    def max_temperature(self): return PropsSI("TMAX", "", 0, "", 0, self.name)

    @property
    def min_pressure(self): return PropsSI("PMIN", "", 0, "", 0, self.name)

    @property
    def max_pressure(self): return PropsSI("PMAX", "", 0, "", 0, self.name)

    @property
    def coolprop_name(self) -> str:
        return self.name  # for single-fluid cases

    def __repr__(self):
        state_desc = ', '.join(f"{k}={getattr(self, '_'+k)}" for k in self._input_pair)
        return f"<Fluid({self.name}): {state_desc}>"

    def __str__(self):
        input_map = {"T": "_T", "P": "_P", "Q": "_X"}

        def get_input_value(k):
            return getattr(self, input_map[k])

        def fmt(val, unit="", precision=4):
            if val is None or val != val:
                return None
            if isinstance(val, (float, int)):
                return f"{val:.{precision}f} {unit}".rstrip()
            return f"{val} {unit}".rstrip()
        

        summary = [
            f"Fluid: {self.name}",
            f"Defined by: {self._input_pair[0]} = {fmt(get_input_value(self._input_pair[0]))}, "
            f"{self._input_pair[1]} = {fmt(get_input_value(self._input_pair[1]))}",
            "",
            "--- Thermodynamic State ---"
        ]

        def add_line(label, value, unit="", precision=4):
            v = fmt(value, unit, precision)
            if v is not None:
                summary.append(f"{label:<30} {v}")

        add_line("Phase:", self.phase)
        add_line("Temperature [K]:", self.T)
        add_line("Pressure [Pa]:", self.P)
        add_line("Quality:", self.X)
        add_line("Density [kg/m³]:", self.density)
        add_line("Enthalpy [J/kg]:", self.enthalpy)
        add_line("Specific Heat [J/kg·K]:", self.cp)
        add_line("Viscosity [Pa·s]:", self.viscosity, precision=6)
        add_line("Thermal Conductivity [W/m·K]:", self.thermal_conductivity, precision=6)
        add_line("Speed of Sound [m/s]:", self.speed_of_sound)
        add_line("Prandtl Number:", self.prandtl)

        summary += ["", "--- Fluid Constants ---"]
        add_line("Molecular Weight [kg/mol]:", self.molecular_weight, precision=6)
        add_line("Critical Temperature [K]:", self.critical_temperature)
        add_line("Critical Pressure [Pa]:", self.critical_pressure)
        add_line("Min Temperature [K]:", self.min_temperature, precision=2)
        add_line("Max Temperature [K]:", self.max_temperature, precision=2)
        add_line("Min Pressure [Pa]:", self.min_pressure, precision=2)
        add_line("Max Pressure [Pa]:", self.max_pressure, precision=2)

        return "\n".join(summary)
            
    def set_state(self, *, T=None, P=None, X=None) -> None:
        """
        Reset the fluid state using exactly two of T, P, X.
        The third, non-input property is cleared.
        """
        count = sum(v is not None for v in [T, P, X])
        if count != 2:
            raise ValueError("Exactly two of T, P, or X must be provided.")

        # Clear all state first
        self._T = None
        self._P = None
        self._X = None

        # Set new input values
        self._set_inputs(T, P, X)
        self._update_state()

