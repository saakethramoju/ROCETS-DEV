from typing import Dict, List
from CoolProp.CoolProp import PropsSI
from .Fluid import Fluid


class Mixture(Fluid):
    def __init__(self, fractions: Dict[str, float], *, fraction_type="mole", T=None, P=None, X=None):
        self._fraction_type = fraction_type.lower()
        if self._fraction_type not in ("mole", "mass"):
            raise ValueError("fraction_type must be either 'mole' or 'mass'.")

        total = sum(fractions.values())
        if total <= 0:
            raise ValueError("Fractions must sum to a positive number.")

        self._fractions = {k: v / total for k, v in fractions.items()}
        self._constituents = list(self._fractions.keys())
        self.name = "&".join(f"{comp}[{self._fractions[comp]}]" for comp in self._constituents)

        super().__init__(self.name, T=T, P=P, X=X)
        self._update_state()

    # --- Mixture-Specific Properties ---

    @property
    def is_mixture(self) -> bool:
        return True

    @property
    def constituents(self) -> List[str]:
        return self._constituents

    @property
    def mole_fractions(self) -> Dict[str, float]:
        if self._fraction_type == "mole":
            return self._fractions
        molar_masses = {c: PropsSI("M", c) for c in self._constituents}
        moles = {c: self._fractions[c] / molar_masses[c] for c in self._constituents}
        total_moles = sum(moles.values())
        return {c: moles[c] / total_moles for c in self._constituents}

    @property
    def mass_fractions(self) -> Dict[str, float]:
        if self._fraction_type == "mass":
            return self._fractions
        molar_masses = {c: PropsSI("M", c) for c in self._constituents}
        masses = {c: self._fractions[c] * molar_masses[c] for c in self._constituents}
        total_mass = sum(masses.values())
        return {c: masses[c] / total_mass for c in self._constituents}

    @property
    def coolprop_name(self) -> str:
        return f"HEOS::{self.name}"

    def _prop(self, key):
        input_map = {"T": "_T", "P": "_P", "Q": "_X"}
        k1, k2 = self._input_pair
        v1 = getattr(self, input_map[k1])
        v2 = getattr(self, input_map[k2])
        try:
            return PropsSI(key, k1, v1, k2, v2, self.coolprop_name)
        except Exception:
            return float("nan")

    def _safe_prop(self, key):
        try:
            return PropsSI(key, "", 0, "", 0, self.coolprop_name)
        except Exception:
            return None

    # Override only the critical properties that may not work for mixtures
    @property
    def molecular_weight(self): return self._safe_prop("M")
    @property
    def critical_temperature(self): return self._safe_prop("TCRIT")
    @property
    def critical_pressure(self): return self._safe_prop("PCRIT")
    @property
    def min_temperature(self): return self._safe_prop("TMIN")
    @property
    def max_temperature(self): return self._safe_prop("TMAX")
    @property
    def min_pressure(self): return self._safe_prop("PMIN")
    @property
    def max_pressure(self): return self._safe_prop("PMAX")

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
            f"Mixture: " + "&".join(f"{comp}[{self.mole_fractions[comp]:.4f}]" for comp in self.constituents),
            f"Defined by: {self._input_pair[0]} = {fmt(get_input_value(self._input_pair[0]))}, "
            f"{self._input_pair[1]} = {fmt(get_input_value(self._input_pair[1]))}",
            "",
            "--- Mole Fractions ---"
        ]

        for comp in self.constituents:
            summary.append(f"{comp:<30} {self.mole_fractions[comp]:.4f}")

        summary.append("")
        summary.append("--- Mass Fractions ---")
        for comp in self.constituents:
            summary.append(f"{comp:<30} {self.mass_fractions[comp]:.4f}")

        summary += ["", "--- Thermodynamic State ---"]

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

    def _set_fractions(self, fractions: Dict[str, float], mode: str):
        if mode not in ("mole", "mass"):
            raise ValueError("Mode must be 'mole' or 'mass'.")
        total = sum(fractions.values())
        if total <= 0:
            raise ValueError(f"{mode.capitalize()} fractions must sum to a positive number.")

        normalized = {k: v / total for k, v in fractions.items()}
        self._fractions = normalized
        self._fraction_type = mode
        self._constituents = list(normalized.keys())

        if mode == "mass":
            molar_masses = {c: PropsSI("M", c) for c in normalized}
            moles = {c: normalized[c] / molar_masses[c] for c in normalized}
            total_moles = sum(moles.values())
            mole_fracs = {c: moles[c] / total_moles for c in normalized}
        else:
            mole_fracs = normalized

        self.name = "&".join(f"{comp}[{mole_fracs[comp]}]" for comp in mole_fracs)
        self._update_state()

    def set_mole_fractions(self, new_fractions): self._set_fractions(new_fractions, "mole")
    def set_mass_fractions(self, new_fractions): self._set_fractions(new_fractions, "mass")


    @mole_fractions.setter
    def mole_fractions(self, value: Dict[str, float]):
        self.set_mole_fractions(value)

    @mass_fractions.setter
    def mass_fractions(self, value: Dict[str, float]):
        self.set_mass_fractions(value)
