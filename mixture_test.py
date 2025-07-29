from Fluid import Fluid
from Components import FluidStateInlet

# Create a mixture: 60% Propane, 40% Ethane (mole basis)
f = Fluid(
    name="C3/C2 Mix",
    mixture={"Propane": 0.6, "Ethane": 0.4},
    fractions_basis="mole",
    rounding=3,
    T=280.0,
    P=2e6
)

# Attach to a boundary inlet port
inlet = FluidStateInlet("FuelInlet", "fuel")
inlet.outflow.fluid = f


def print_state(label):
    print(f"\n--- {label} ---")
    print("T [K]:", f.T)
    print("P [Pa]:", f.P)
    print("X:", f.X)
    print("Phase:", f.phase)
    print("Density [kg/m³]:", f.density)
    print("Enthalpy [J/kg]:", f.enthalpy)
    print("Cp [J/kg·K]:", f.cp)
    print("Viscosity [Pa·s]:", f.viscosity)
    print("Speed of sound [m/s]:", f.speed_of_sound)
    print("Thermal conductivity [W/m·K]:", f.thermal_conductivity)
    print("Molecular weight [kg/kmol]:", f.molecular_weight)


# 🔍 Initial state
#print_state("Initial state")

# 🔥 Raise temperature through Fluid
f.T = 300.0
print(inlet)
#print_state("After raising T via Fluid")

# 📈 Raise pressure through port
inlet.outflow.P = 3e6
print(inlet)
#print_state("After raising P via Port")

# 💧 Set vapor quality through port (2 of 3: P and X)
inlet.outflow.fluid.set_state(P=3e6, X=0.5)
#print_state("After setting quality X = 0.5 via Port")

#print("-------------------------")
#print(f)


from __future__ import annotations

from typing import Dict, Optional, Iterable, Tuple
from CoolProp.CoolProp import PropsSI
from CoolProp import CoolProp


def _normalize(values: Iterable[float]) -> Tuple[float, ...]:
    vals = tuple(float(v) for v in values)
    total = sum(vals)
    if total <= 0:
        raise ValueError("Fractions must sum to a positive value.")
    return tuple(v / total for v in vals)


class Fluid:
    def __init__(
        self,
        name: str,
        *,
        T=None,
        P=None,
        X=None,
        mixture: Optional[Dict[str, float]] = None,
        fractions_basis: str = "mole",   # "mole" or "mass"
        rounding: int = 4,               # decimals for mixture label like HEOS::A[0.6]&B[0.4]
        backend: str = "HEOS",
    ):
        """
        Fluid state for a pure fluid or mixture using any two of (T, P, X).

        Parameters
        ----------
        name : str
            User-defined label (e.g. "Fuel", "MyMixture").
            This will appear in tables and component printouts.
        mixture : dict[str, float], optional
            Constituent → fraction mapping. Interpreted as mole fractions by default;
            set fractions_basis="mass" to supply mass fractions.
        fractions_basis : {"mole", "mass"}
            Basis for the `mixture` fractions if provided.
        rounding : int
            Number of decimals used when generating the internal mixture identifier.
        backend : str
            CoolProp backend for AbstractState (e.g. "HEOS").
        """
        self.backend = backend
        self._round = int(rounding)
        self._state = {}         # stores latest canonical T, P (and last-set X if applicable)
        self._as = None          # AbstractState when we build mixtures from a dict
        self.name = name         # user-visible label (for display)
        self.identifier = None   # internal identifier string
        self._fluid_id = None    # used for PropsSI

        if mixture is not None:
            # Build from components dict -> AbstractState, with optional mass→mole conversion.
            comps = sorted(mixture.keys())  # stable ordering
            fracs_in = [mixture[c] for c in comps]

            if fractions_basis not in ("mole", "mass"):
                raise ValueError("fractions_basis must be 'mole' or 'mass'")

            if fractions_basis == "mole":
                x = _normalize(fracs_in)
            else:
                # Convert mass fractions to mole fractions: x_i ∝ w_i / M_i
                Ms = []
                for comp in comps:
                    as_i = CoolProp.AbstractState(self.backend, comp)
                    Ms.append(as_i.molar_mass())  # kg/mol
                n_unnorm = [w / M for w, M in zip(fracs_in, Ms)]
                x = _normalize(n_unnorm)

            spec = "&".join(comps)
            try:
                self._as = CoolProp.AbstractState(self.backend, spec)
                self._as.set_mole_fractions(x)
            except Exception as e:
                raise ValueError(f"Failed to create AbstractState for mixture '{spec}': {e}")

            # Internal mixture ID for PropsSI and caching
            self.identifier = (
                f"{self.backend}::" + "&".join(f"{c}[{xi:.{self._round}f}]" for c, xi in zip(comps, x))
            )
            self._fluid_id = self.identifier
        else:
            # Pure fluid or already-formatted CoolProp mixture string
            self.identifier = name
            self._fluid_id = name

        # Build initial state if provided
        if sum(v is not None for v in (T, P, X)) > 0:
            self._build_state(T, P, X)

    # ------------------------- internal build/update -------------------------

    def _build_state(self, T, P, X):
        # Count how many inputs are provided
        count = sum(x is not None for x in (T, P, X))
        if count != 2:
            raise ValueError("You must provide exactly two of: T, P, X")

        # If this is a retry or fresh build, always clear the previous state
        self._state.clear()

        if self._as is not None:
            try:
                if T is not None and P is not None:
                    self._as.update(CoolProp.PT_INPUTS, float(P), float(T))
                elif P is not None and X is not None:
                    self._as.update(CoolProp.PQ_INPUTS, float(P), float(X))
                elif T is not None and X is not None:
                    self._as.update(CoolProp.QT_INPUTS, float(X), float(T))
                else:
                    raise AssertionError("Unreachable")
                # cache canonical T, P; remember last X if applicable
                self._state["T"] = self._as.T()
                self._state["P"] = self._as.p()
                if X is not None:
                    self._state["X"] = X
                else:
                    self._state.pop("X", None)
                return self
            except Exception as e:
                raise ValueError(f"Failed to initialize state for '{self.name}' using (T, P, X): {e}")

        # Pure fluid fallback
        try:
            if T is not None and P is not None:
                self._state["T"] = float(T)
                self._state["P"] = float(P)
                self._state.pop("X", None)
            elif P is not None and X is not None:
                self._state["P"] = float(P)
                self._state["T"] = PropsSI("T", "P", float(P), "Q", float(X), self._fluid_id)
                self._state["X"] = float(X)
            elif T is not None and X is not None:
                self._state["T"] = float(T)
                self._state["P"] = PropsSI("P", "T", float(T), "Q", float(X), self._fluid_id)
                self._state["X"] = float(X)
            return self
        except Exception as e:
            raise ValueError(f"Failed to initialize fluid '{self.name}' with given inputs (T={T}, P={P}, X={X}): {e}")

    def _update_single(self, key: str, val: float):
        """
        Update the state when only one of T/P/X changes.
        Uses whichever of the other two is already known; otherwise raises.
        """
        known_T = self._state.get("T")
        known_P = self._state.get("P")
        known_X = self._state.get("X", None)

        if key == "T":
            if known_P is not None:
                return self._build_state(val, known_P, None)
            if known_X is not None:
                return self._build_state(val, None, known_X)
        elif key == "P":
            if known_T is not None:
                return self._build_state(known_T, val, None)
            if known_X is not None:
                return self._build_state(None, val, known_X)
        elif key == "X":
            if known_T is not None:
                return self._build_state(known_T, None, val)
            if known_P is not None:
                return self._build_state(None, known_P, val)

        raise ValueError(
            f"Cannot update {key} alone: need one of the other variables already known."
        )

    # ------------------------- user-facing mutation API -------------------------
    # ------------------------- internal update from current _state -------------------------

    def _update_fluid(self):
        T = self._state.get("T")
        P = self._state.get("P")
        X = self._state.get("X", None)

        # Must have exactly two of T, P, X to define a valid state
        if sum(v is not None for v in (T, P, X)) != 2:
            raise ValueError("Exactly two of (T, P, X) must be provided to set the state.")

        # Clear the current state to ensure a clean rebuild
        self._state.clear()

        try:
            if self._as is not None:
                if T is not None and P is not None:
                    self._as.update(CoolProp.PT_INPUTS, float(P), float(T))
                elif P is not None and X is not None:
                    self._as.update(CoolProp.PQ_INPUTS, float(P), float(X))
                elif T is not None and X is not None:
                    self._as.update(CoolProp.QT_INPUTS, float(X), float(T))
                # Sync stored state from CoolProp
                self._state["T"] = self._as.T()
                self._state["P"] = self._as.p()
                if X is not None:
                    self._state["X"] = X
                else:
                    self._state.pop("X", None)
            else:
                # Pure fluid fallback using PropsSI
                if T is not None and P is not None:
                    self._state["T"] = float(T)
                    self._state["P"] = float(P)
                    self._state.pop("X", None)
                elif P is not None and X is not None:
                    self._state["P"] = float(P)
                    self._state["T"] = PropsSI("T", "P", float(P), "Q", float(X), self._fluid_id)
                    self._state["X"] = float(X)
                elif T is not None and X is not None:
                    self._state["T"] = float(T)
                    self._state["P"] = PropsSI("P", "T", float(T), "Q", float(X), self._fluid_id)
                    self._state["X"] = float(X)
        except Exception as e:
            raise ValueError(f"Failed to update fluid '{self.name}' with state (T={T}, P={P}, X={X}): {e}")


    # ------------------------- final cleanup -------------------------
    
    def set_state(self, *, T=None, P=None, X=None):
        """
        Mutate this Fluid in place using one or two of (T, P, X).
        If one is provided, the others must be cached. If two are provided, they override cached values.
        """
        count = sum(v is not None for v in (T, P, X))
        if count == 0:
            return self
        if count == 1:
            # update using cached values
            if T is not None:
                return self._update_single("T", float(T))
            if P is not None:
                return self._update_single("P", float(P))
            if X is not None:
                return self._update_single("X", float(X))
        elif count == 2:
            # clear the previous state and rebuild from scratch
            self._state.clear()
            return self._build_state(T, P, X)
        else:
            # all three provided — prefer T+P
            self._state.clear()
            return self._build_state(T, P, None)


    # Short aliases so ports can call .T/.P/.X if they want
    @property
    def T(self): return self._state.get("T")

    @T.setter
    def T(self, v): self.set_state(T=v)

    @property
    def P(self): return self._state.get("P")

    @P.setter
    def P(self, v): self.set_state(P=v)

    @property
    def X(self): return self._state.get("X", None)

    @X.setter
    def X(self, v): self.set_state(X=v)

    # Backwards-compatible names
    @property
    def temperature(self): return self.T

    @temperature.setter
    def temperature(self, v): self.T = v

    @property
    def pressure(self): return self.P

    @pressure.setter
    def pressure(self, v): self.P = v

    @property
    def quality(self):
        # If we know we're two-phase, or if Q is defined, return it; else None
        if self._as is not None:
            try:
                x = self._as.Q()
                if 0.0 <= x <= 1.0:
                    return x
            except Exception:
                pass
            return self._state.get("X", None)
        try:
            x = PropsSI("Q", "T", self.T, "P", self.P, self._fluid_id)
            if 0.0 <= x <= 1.0:
                return x
        except Exception:
            pass
        return self._state.get("X", None)

    # ------------------------- properties -------------------------

    @property
    def phase(self) -> str:
        try:
            if self._as is not None:
                code = int(self._as.phase())
            else:
                code = int(PropsSI("Phase", "T", self.T, "P", self.P, self._fluid_id))
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
            CoolProp.iphase_not_imposed: "not imposed",
        }
        return phase_map.get(code, f"unrecognized({code})")

    @property
    def density(self):
        if self._as is not None:
            return self._as.rhomass()
        return PropsSI("D", "T", self.T, "P", self.P, self._fluid_id)

    @property
    def enthalpy(self):
        if self._as is not None:
            return self._as.hmass()
        return PropsSI("H", "T", self.T, "P", self.P, self._fluid_id)

    @property
    def viscosity(self):
        if self._as is not None:
            return self._as.viscosity()
        return PropsSI("V", "T", self.T, "P", self.P, self._fluid_id)

    @property
    def cp(self):
        if self._as is not None:
            return self._as.cpmass()
        # Keep using your original symbol. (CoolProp's explicit key is "Cpmass".)
        return PropsSI("C", "T", self.T, "P", self.P, self._fluid_id)

    @property
    def saturation_pressure(self):
        # Bubble line at current T (Q=0) — may not exist for all mixtures
        return PropsSI("P", "T", self.T, "Q", 0, self._fluid_id)

    @property
    def saturation_temperature(self):
        # Bubble line at current P (Q=0)
        return PropsSI("T", "P", self.P, "Q", 0, self._fluid_id)

    @property
    def critical_temperature(self):
        return PropsSI("TCRIT", self._fluid_id)

    @property
    def critical_pressure(self):
        return PropsSI("PCRIT", self._fluid_id)

    @property
    def min_temperature(self):
        return PropsSI("Tmin", self._fluid_id)

    @property
    def max_temperature(self):
        return PropsSI("Tmax", self._fluid_id)

    @property
    def min_pressure(self):
        return PropsSI("Pmin", self._fluid_id)

    @property
    def max_pressure(self):
        return PropsSI("Pmax", self._fluid_id)
        
    @property
    def is_mixture(self) -> bool:
        """
        True if this Fluid is a mixture created via AbstractState.
        """
        return self._as is not None


    @property
    def constituents(self) -> Optional[list[str]]:
        """
        List of constituent fluid names.
        Returns None for pure fluids.
        """
        if not self._as:
            return None
        names = self._as.fluid_names()
        if isinstance(names, str):
            return names.split("&")
        if isinstance(names, list):
            return names
        return None

    @property
    def mole_fractions(self) -> Optional[Dict[str, float]]:
        """
        Dict of constituent → mole fraction.
        Returns None for pure fluids.
        """
        if not self._as:
            return None
        return dict(zip(self.constituents, self._as.get_mole_fractions()))

    @property
    def mass_fractions(self) -> Optional[Dict[str, float]]:
        """
        Dict of constituent → mass fraction.
        Returns None for pure fluids.
        """
        if not self._as:
            return None
        return dict(zip(self.constituents, self._as.get_mass_fractions()))


    @property
    def mixture(self) -> Dict[str, float]:
        """
        Alias for mole_fractions.
        """
        return self.mole_fractions
    

    @property
    def molecular_weight(self) -> float:
        """
        Molecular weight [kg/mol] for the fluid or mixture.
        For mixtures, this is the mixture-average molar mass from CoolProp.
        """
        if self._as is not None:
            return self._as.molar_mass()  # kg/mol
        # Try PropsSI with a static call; if unavailable, use a temporary AbstractState
        try:
            return PropsSI("M", self._fluid_id)  # kg/mol
        except Exception:
            return CoolProp.AbstractState(self.backend, self._fluid_id).molar_mass()

    @property
    def thermal_conductivity(self) -> float:
        """
        Thermal conductivity k [W/(m·K)].
        """
        if self._as is not None:
            return self._as.conductivity()
        return PropsSI("L", "T", self.T, "P", self.P, self._fluid_id)

    @property
    def speed_of_sound(self) -> float:
        """
        Speed of sound a [m/s].
        """
        if self._as is not None:
            return self._as.speed_sound()
        return PropsSI("A", "T", self.T, "P", self.P, self._fluid_id)

    @property
    def prandtl_number(self) -> float:
        """
        Prandtl number Pr = (cp * mu) / k  [dimensionless].
        Uses mass-based cp (J/kg·K), dynamic viscosity mu (Pa·s), and k (W/m·K).
        """
        if self._as is not None:
            cp = self._as.cpmass()         # J/kg-K
            mu = self._as.viscosity()      # Pa·s
            k  = self._as.conductivity()   # W/m-K
        else:
            cp = PropsSI("C", "T", self.T, "P", self.P, self._fluid_id)
            mu = PropsSI("V", "T", self.T, "P", self.P, self._fluid_id)
            k  = PropsSI("L", "T", self.T, "P", self.P, self._fluid_id)
        return cp * mu / k




    # ------------------------- convenience & repr -------------------------

    def summary(self) -> dict:
        def safe(fn):
            try:
                return fn()
            except Exception:
                return "N/A"

        return {
            "Fluid": self.name,
            "Temperature (K)": self.T,
            "Pressure (Pa)": self.P,
            "Vapor Quality": self.quality,
            "Phase": self.phase,
            "Density (kg/m³)": safe(lambda: self.density),
            "Enthalpy (J/kg)": safe(lambda: self.enthalpy),
            "Viscosity (Pa·s)": safe(lambda: self.viscosity),
            "Cp (J/kg·K)": safe(lambda: self.cp),
            "Saturation Pressure (Pa)": safe(lambda: self.saturation_pressure),
            "Saturation Temperature (K)": safe(lambda: self.saturation_temperature),
            "Critical Temperature (K)": safe(lambda: self.critical_temperature),
            "Critical Pressure (Pa)": safe(lambda: self.critical_pressure),
        }

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.summary().items())

    def __repr__(self):
        return f"<Fluid {self.name} | T={self.T} K, P={self.P} Pa, X={self.quality}>"

    def is_defined(self) -> bool:
        """
        True if the state is well-defined: at least two of (T, P, X) are known.
        """
        # quality() may fail outside 2-phase; rely on what we have stored
        return sum(v is not None for v in (self.T, self.P, self._state.get("X", None))) >= 2
            


