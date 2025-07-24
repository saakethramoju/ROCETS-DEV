from CoolProp.CoolProp import PropsSI
from CoolProp import CoolProp


class Fluid:
    def __init__(self, name: str, *, T=None, P=None, X=None):
        """
        Create a fluid state for a supported pure substance using any two of:
        - T: temperature [K]
        - P: pressure [Pa]
        - X: vapor quality [0–1] (only for two-phase fluids)
        """
        self.name = name
        self._state = {}
        self._build_state(name, T, P, X)

    def _build_state(self, name, T, P, X):
        if sum(x is not None for x in [T, P, X]) != 2:
            raise ValueError("You must provide exactly two of: T, P, X")

        try:
            if T is not None and P is not None:
                self._state["T"] = T
                self._state["P"] = P
            elif P is not None and X is not None:
                self._state["P"] = P
                self._state["T"] = PropsSI("T", "P", P, "Q", X, name)
            elif T is not None and X is not None:
                self._state["T"] = T
                self._state["P"] = PropsSI("P", "T", T, "Q", X, name)
        except Exception as e:
            raise ValueError(f"Failed to initialize fluid '{name}' with given inputs: {e}")

    @property
    def temperature(self):
        return self._state.get("T")

    @property
    def pressure(self):
        return self._state.get("P")

    @property
    def quality(self):
        try:
            x = PropsSI("Q", "T", self.temperature, "P", self.pressure, self.name)
            if 0.0 <= x <= 1.0:
                return x
        except Exception:
            pass
        return None

    @property
    def phase(self) -> str:
        try:
            code = int(PropsSI("Phase", "T", self.temperature, "P", self.pressure, self.name))
        except Exception:
            return "unknown"
        phase_map = {
            CoolProp.iphase_liquid: "liquid",
            CoolProp.iphase_supercritical: "supercritical",
            CoolProp.iphase_supercritical_gas: "supercritical gas",
            CoolProp.iphase_supercritical_liquid: "supercritical liquid",
            CoolProp.iphase_critical_point: "critical point",
            CoolProp.iphase_gas: "gas",
            CoolProp.iphase_twophase: "two-phase",
            CoolProp.iphase_unknown: "unknown",
            CoolProp.iphase_not_imposed: "not imposed"
        }
        return phase_map.get(code, f"unrecognized({code})")

    @property
    def density(self):
        return PropsSI("D", "T", self.temperature, "P", self.pressure, self.name)

    @property
    def enthalpy(self):
        return PropsSI("H", "T", self.temperature, "P", self.pressure, self.name)

    @property
    def viscosity(self):
        return PropsSI("V", "T", self.temperature, "P", self.pressure, self.name)

    @property
    def cp(self):
        return PropsSI("C", "T", self.temperature, "P", self.pressure, self.name)

    @property
    def saturation_pressure(self):
        return PropsSI("P", "T", self.temperature, "Q", 0, self.name)

    @property
    def saturation_temperature(self):
        return PropsSI("T", "P", self.pressure, "Q", 0, self.name)

    @property
    def critical_temperature(self):
        return PropsSI("TCRIT", self.name)

    @property
    def critical_pressure(self):
        return PropsSI("PCRIT", self.name)
    

    @property
    def min_temperature(self):
        """Minimum temperature CoolProp supports for this fluid."""
        return PropsSI("Tmin", self.name)

    @property
    def max_temperature(self):
        """Maximum temperature CoolProp supports for this fluid."""
        return PropsSI("Tmax", self.name)

    @property
    def min_pressure(self):
        """Minimum pressure CoolProp supports for this fluid."""
        return PropsSI("Pmin", self.name)

    @property
    def max_pressure(self):
        """Maximum pressure CoolProp supports for this fluid."""
        return PropsSI("Pmax", self.name)


    def summary(self) -> dict:
        def safe(prop_fn):
            try:
                return prop_fn()
            except:
                return "N/A"

        return {
            "Fluid": self.name,
            "Temperature (K)": self.temperature,
            "Pressure (Pa)": self.pressure,
            "Vapor Quality": self.quality,
            "Phase": self.phase,
            "Density (kg/m³)": safe(self.density),
            "Enthalpy (J/kg)": safe(self.enthalpy),
            "Viscosity (Pa·s)": safe(self.viscosity),
            "Cp (J/kg·K)": safe(self.cp),
            "Saturation Pressure (Pa)": safe(self.saturation_pressure),
            "Saturation Temperature (K)": safe(self.saturation_temperature),
            "Critical Temperature (K)": safe(self.critical_temperature),
            "Critical Pressure (Pa)": safe(self.critical_pressure),
        }

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.summary().items())

    def __repr__(self):
        return f"<Fluid {self.name} | T={self.temperature} K, P={self.pressure} Pa, X={self.quality}>"
