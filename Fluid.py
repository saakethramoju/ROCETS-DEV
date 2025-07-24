from pyfluids import Fluid as PF_Fluid, FluidsList, Input
from CoolProp.CoolProp import PropsSI


class Fluid:
    def __init__(self, name: str, *, T=None, P=None, X=None):
        """
        Create a fluid state given any two of:
        - T: temperature [K]
        - P: pressure [Pa]
        - X: vapor quality [0–1]
        """
        self.name = name
        self._fluid = self._build_fluid(name, T, P, X)

    def _build_fluid(self, name, T, P, X):
        # Get PyFluids fluid enum
        fluid_enum = getattr(FluidsList, name.capitalize(), None)
        if not fluid_enum:
            raise ValueError(f"Unsupported fluid: '{name}'")

        fluid = PF_Fluid(fluid_enum)

        if T is not None and P is not None:
            return fluid.with_state(Input.temperature(T), Input.pressure(P))

        elif P is not None and X is not None:
            return fluid.two_phase_point_at_pressure(P, X)

        elif T is not None and X is not None:
            # Fallback: Use CoolProp to compute Psat(T)
            try:
                Psat = PropsSI("P", "T", T, "Q", X, name)
            except Exception as e:
                raise ValueError(f"Failed to calculate saturation pressure for {name} at T={T} K: {e}")
            return fluid.two_phase_point_at_pressure(Psat, X)

        else:
            raise ValueError("You must provide exactly two of: T, P, X")

    # --- Properties ---
    @property
    def phase(self): return self._fluid.phase.name

    @property
    def temperature(self): return self._fluid.temperature

    @property
    def pressure(self): return self._fluid.pressure

    @property
    def quality(self): return self._fluid.quality

    @property
    def density(self): return self._fluid.density

    @property
    def enthalpy(self): return self._fluid.enthalpy

    @property
    def viscosity(self): return self._fluid.dynamic_viscosity

    @property
    def cp(self): return self._fluid.specific_heat

    def get_saturation_pressure(self, T: float) -> float:
        """Return Psat [Pa] at given temperature [K] using CoolProp, if below critical."""
        Tcrit = PropsSI("TCRIT", self.name)
        if T >= Tcrit:
            raise ValueError(f"Temperature {T:.2f} K exceeds critical temperature ({Tcrit:.2f} K) — no saturation pressure exists.")
        try:
            return PropsSI("P", "T", T, "Q", 0, self.name)
        except Exception as e:
            raise ValueError(f"Failed to get Psat for {self.name} at T={T} K: {e}")

    def get_saturation_temperature(self, P: float) -> float:
        """Return Tsat [K] at given pressure [Pa] using CoolProp, if below critical."""
        Pcrit = PropsSI("PCRIT", self.name)
        if P >= Pcrit:
            raise ValueError(f"Pressure {P:.2f} Pa exceeds critical pressure ({Pcrit:.2f} Pa) — no saturation temperature exists.")
        try:
            return PropsSI("T", "P", P, "Q", 0, self.name)
        except Exception as e:
            raise ValueError(f"Failed to get Tsat for {self.name} at P={P} Pa: {e}")

    
    @property
    def saturation_temperature(self) -> float:
        """
        Return Tsat [K] at current pressure, or raise if above critical.
        """
        if self.pressure is None:
            raise AttributeError("Pressure is not defined for this fluid state.")
        
        Pcrit = PropsSI("PCRIT", self.name)
        if self.pressure >= Pcrit:
            raise ValueError(f"Pressure {self.pressure:.0f} Pa exceeds critical pressure for {self.name} ({Pcrit:.0f} Pa) — no saturation temperature exists.")

        return PropsSI("T", "P", self.pressure, "Q", 0, self.name)

    @property
    def saturation_pressure(self) -> float:
        """Return Psat [Pa] using current temperature, or raise if above critical."""
        if self.temperature is None:
            raise AttributeError("Temperature is not defined.")
        return self.get_saturation_pressure(self.temperature)

    @property
    def saturation_temperature(self) -> float:
        """Return Tsat [K] using current pressure, or raise if above critical."""
        if self.pressure is None:
            raise AttributeError("Pressure is not defined.")
        return self.get_saturation_temperature(self.pressure)
        
    @property
    def critical_temperature(self) -> float:
        """Return critical temperature [K] for the fluid."""
        return PropsSI("TCRIT", self.name)

    @property
    def critical_pressure(self) -> float:
        """Return critical pressure [Pa] for the fluid."""
        return PropsSI("PCRIT", self.name)


    def summary(self) -> dict:
        summary = {
            "Fluid": self.name,
            "Phase": self.phase,
            "Temperature (K)": self.temperature,
            "Pressure (Pa)": self.pressure,
            "Vapor Quality": self.quality,
            "Density (kg/m³)": self.density,
            "Enthalpy (J/kg)": self.enthalpy,
            "Viscosity (Pa·s)": self.viscosity,
            "Cp (J/kg·K)": self.cp
        }

        # Add saturation values only if they exist (check via property)
        try:
            summary["Saturation Pressure (Pa)"] = self.saturation_pressure
        except Exception:
            summary["Saturation Pressure (Pa)"] = "N/A"

        try:
            summary["Saturation Temperature (K)"] = self.saturation_temperature
        except Exception:
            summary["Saturation Temperature (K)"] = "N/A"

        return summary


    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.summary().items())

