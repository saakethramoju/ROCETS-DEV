from pyfluids import FluidsList as pyFluidsList
from pyfluids import Fluid as pyFluid
from pyfluids import Mixture as pyMixture
from pyfluids import Input as pyInput
from scipy.optimize import root_scalar
from typing import List, Union, Dict, Tuple
import numpy as np

class Fluid:
    def __init__(
        self,
        fluid: Union[str, Dict[str, float]],   # str = pure fluid, dict = mixture
        basis: str = "mole",                   # 'mole' or 'mass'
        P: float = None,
        h: float = None,
        T: float = None,
        Q: float = None
    ):
        
        valid_fluids = Fluid.get_available_fluids()

        if isinstance(fluid, str):
            if fluid not in valid_fluids:
                raise ValueError(f"Invalid fluid '{fluid}'. Use Fluid.show_available_fluids() to check valid fluid names.")
        elif isinstance(fluid, dict):
            for f in fluid.keys():
                if f not in valid_fluids:
                    raise ValueError(f"Invalid fluid '{fluid}'. Use Fluid.show_available_fluids() to check valid fluid names.")
        else:
            raise TypeError("fluid must be either string (pure) or dictionary (mixture)")
        
        if isinstance(fluid, str):
            self._fluids = [fluid]
            self._mole_fractions = np.array([1.0])
            self._mass_fractions = np.array([1.0])
        elif isinstance(fluid, dict):
            self._fluids = list(fluid.keys())
            fractions = list(fluid.values())
            if basis == "mole":
                self._mole_fractions = np.array(fractions, dtype=float)
                self._mass_fractions = Fluid.mole_to_mass(self._fluids, fractions)
            elif basis == "mass":
                self._mole_fractions = Fluid.mass_to_mole(self._fluids, fractions)
                self._mass_fractions = np.array(fractions, dtype=float)
            else:
                raise ValueError("basis must be 'mole' or 'mass'!")
        else:
            raise TypeError("fluid must be either string (pure) or dictionary (mixture)")
        
        if len(self._fluids) > 1:
            self._mixture = True
        else:
            self._mixture = False
                

        if P is not None and h is not None:
            self._P, self._h = P, h

        elif P is not None and T is not None:
            self._P = P
            if self._mixture:
                st = pyMixture([pyFluidsList[f] for f in self._fluids], self._mass_fractions)
            else:
                st = pyFluid(pyFluidsList[self._fluids[0]])
            self._h = st.with_state(pyInput.temperature(T), pyInput.pressure(P)).enthalpy

        elif P is not None and Q is not None:
            self._P = P
            if self._mixture:
                st = pyMixture([pyFluidsList[f] for f in self._fluids], self._mass_fractions)
            else:
                st = pyFluid(pyFluidsList[self._fluids[0]])
            self._h = st.with_state(pyInput.quality(Q), pyInput.pressure(P)).enthalpy

        elif T is not None and Q is not None:
            if self._mixture:
                st = pyMixture([pyFluidsList[f] for f in self._fluids], self._mass_fractions)
            else:
                st = pyFluid(pyFluidsList[self._fluids[0]])
            self._P = st.with_state(pyInput.quality(Q), pyInput.temperature(T)).pressure
            self._h = st.with_state(pyInput.quality(Q), pyInput.temperature(T)).enthalpy

        else:
            raise LookupError("Please provide at least two thermodynamic properties!")
        
    
        if self._mixture:
            st = pyMixture([pyFluidsList[f] for f in self._fluids],
                                    self._mass_fractions)
            T, Q = Fluid.get_temperature_and_quality(st, self._P, self._h)
            if Q != 1.0 or Q != 0.0:
                self._pyfluid = st.with_state(pyInput.quality(Q), pyInput.pressure(self._P))
            else:
                self._pyfluid = st.with_state(pyInput.temperature(T), pyInput.pressure(self._P))
        else:
            self._pyfluid = pyFluid(pyFluidsList[self._fluids[0]]).with_state(pyInput.pressure(self._P), pyInput.enthalpy(self._h))

            
    @property
    def species(self):
        return self._fluids

    @property
    def mole_fractions(self) -> dict:
        """Mole fractions as {fluid_name: value}"""
        return {
            f: float(x) 
            for f, x in zip(self._fluids, self._mole_fractions)
        }

    @property
    def mass_fractions(self) -> dict:
        """Mass fractions as {fluid_name: value}"""
        mf = Fluid.mole_to_mass(self._fluids, self._mole_fractions)
        return {
            f: float(x) 
            for f, x in zip(self._fluids, mf)
        }
    
    @property
    def pressure(self) -> float:
        """Absolute pressure (Pa)"""
        return self._P
    
    @property
    def enthalpy(self) -> float:
        """Mass specifc enthalpy (J/kg)"""
        return self._h
    
    @property
    def temperature(self) -> float:
        """Absoluate temperature (K)"""
        return self._pyfluid.temperature
    
    @property
    def phase(self) -> str:
        """CoolProp fluid phase"""
        return self._pyfluid.phase.name
    
    @property
    def compressibility(self) -> float:
        """Compressibility factor (dimensionless)"""
        return self._pyfluid.compressibility
    
    @property
    def conductivity(self) -> float:
        """Thermal conductivity (W/m-K)"""
        return self._pyfluid.conductivity

    @property
    def critical_pressure(self) -> float:
        """Absolute pressure at the critical point (Pa)"""
        return self._pyfluid.critical_pressure

    @property
    def critical_temperature(self) -> float:
        """Temperature at the critical point"""
        return self._pyfluid.critical_temperature

    @property
    def density(self) -> float:
        """Mass density (kg/m^3)"""
        return self._pyfluid.density
    
    @property
    def dynamic_viscosity(self) -> float:
        """Dynamic viscosity (Pa-s)"""
        return self._pyfluid.dynamic_viscosity

    @property
    def entropy(self) -> float:
        """Mass specific entropy (J/kg-K)"""
        return self._pyfluid.entropy

    @property
    def freezing_temperature(self) -> float:
        """Freezing point temperature (K)"""
        return self._pyfluid.freezing_temperature

    @property
    def internal_energy(self) -> float:
        """Mass specific internal energy (J/kg)"""
        return self._pyfluid.internal_energy

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity (m^2/s)"""
        return self._pyfluid.kinematic_viscosity

    @property
    def max_pressure(self) -> float:
        """Maximum valid pressure (Pa)"""
        return self._pyfluid.max_pressure

    @property
    def max_temperature(self) -> float:
        """Maximum valid temperature (K)"""
        return self._pyfluid.max_temperature
    
    @property
    def min_pressure(self) -> float:
        """Minimum valid pressure (Pa)"""
        return self._pyfluid.min_pressure

    @property
    def min_temperature(self) -> float:
        """Minimum valid temperature (K)"""
        return self._pyfluid.min_temperature

    @property
    def molar_mass(self) -> float:
        """Molar mass (kg/mol)"""
        return self._pyfluid.molar_mass

    @property
    def prandtl(self) -> float:
        """Prandtl number (dimensionless)"""
        return self._pyfluid.prandtl
    
    @property
    def speed_of_sound(self) -> float:
        """Speed of sound (m/s)"""
        return self._pyfluid.sound_speed

    @property
    def specific_heat(self) -> float:
        """Specific heat at constant pressure (J/kg-K)"""
        return self._pyfluid.specific_heat

    @property
    def specific_volume(self) -> float:
        """Mass specific volume (m^3/kg)."""
        return self._pyfluid.specific_volume

    @property
    def surface_tension(self) -> float:
        """Surface tension (N/m)"""
        return self._pyfluid.surface_tension

    @property
    def triple_pressure(self) -> float:
        """Triple point pressure (Pa)"""
        return self._pyfluid.triple_pressure
    
    @property
    def triple_temperature(self) -> float:
        """Triple point temperature (K)"""
        return self._pyfluid.triple_temperature
    
    @property
    def is_mixture(self) -> bool:
        """True if the fluid is a mixture"""
        return self._mixture
    
    @property
    def quality(self) -> float:
        """Vapor quality (0-1)"""
        h_liquid = self._pyfluid.with_state(pyInput.pressure(self._P), pyInput.quality(0.0)).enthalpy
        h_vapor = self._pyfluid.with_state(pyInput.pressure(self._P), pyInput.quality(1.0)).enthalpy
        if self._h >= h_vapor:
            return 1.0
        elif self._h <= h_liquid:
            return 0.0
        else:
            return self._pyfluid.quality
        
    @property
    def saturation_temperature(self) -> float:
        """Saturation temperature for fluid's pressure (K)"""
        try:
            return self._pyfluid.with_state(pyInput.pressure(self._P), pyInput.quality(1.0)).temperature
        except:
            raise ValueError("Cannot access saturation temperature for this pressure!")

    def _safe(self, value, fmt=".3e"):
        """Return formatted value or 'N/A' if None"""
        if value is None:
            return "N/A"
        try:
            return f"{value:{fmt}}"
        except Exception:
            return str(value)

    def __str__(self):
        def format_dict(d: dict, decimals=3):
            return {k: round(v, decimals) for k, v in d.items()}

        rows = [
            ("Fluid(s)", ", ".join(self._fluids)),
            ("Mole fractions", format_dict(self.mole_fractions, 3)),
            ("Mass fractions", format_dict(self.mass_fractions, 3)),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e")),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe(self.density, ".3f")),
            ("Quality", self._safe(self.quality, ".3f")),
            ("Internal energy [J/kg]", self._safe(self.internal_energy, ".3e")),
            ("Enthalpy [J/kg]", self._safe(self.enthalpy, ".3e")),
            ("Entropy [J/kg-K]", self._safe(self.entropy, ".3e")),
            ("Dynamic viscosity [Pa·s]", self._safe(self.dynamic_viscosity, ".3e")),
            ("Conductivity [W/m-K]", self._safe(self.conductivity, ".3f")),
            ("Saturation temperature [K]", self._safe(self.saturation_temperature, ".2f")),
            ("Molar mass [kg/mol]", self._safe(self.molar_mass, ".6f")),
        ]

        width = max(len(r[0]) for r in rows)
        return "\n".join(f"{key:<{width}} : {val}" for key, val in rows)


       

    # ---- Static utilities ---- #
    @staticmethod
    def mole_to_mass(fluids: List[str], mole_fractions: List[float]):
        if not np.isclose(sum(mole_fractions), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0")
        mole_fractions = np.asarray(mole_fractions, dtype=float)
        molar_masses = np.array([pyFluid(pyFluidsList[f]).molar_mass for f in fluids])
        m_bar = np.dot(mole_fractions, molar_masses)
        return mole_fractions * molar_masses / m_bar

    @staticmethod
    def mass_to_mole(fluids: List[str], mass_fractions: List[float]):
        if not np.isclose(sum(mass_fractions), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0")
        mass_fractions = np.asarray(mass_fractions, dtype=float)
        molar_masses = np.array([pyFluid(pyFluidsList[f]).molar_mass for f in fluids])
        inv = mass_fractions / molar_masses
        return inv / inv.sum()
    

    @staticmethod
    def get_temperature_and_quality(fluid: pyFluid, P:float, target_enthalpy:float) -> Tuple[str]:
        h_liquid = fluid.with_state(pyInput.quality(0), pyInput.pressure(P)).enthalpy
        h_vapor = fluid.with_state(pyInput.quality(1), pyInput.pressure(P)).enthalpy
        h = target_enthalpy
        if h_liquid <= h <= h_vapor:
            Q = (h - h_liquid) / (h_vapor - h_liquid)
            T = fluid.with_state(pyInput.quality(Q), pyInput.pressure(P)).temperature
        else:
            def residual(T):
                try:
                    st = fluid.with_state(pyInput.temperature(T), pyInput.pressure(P))
                    return st.enthalpy - h
                except:
                    return 1e13
            
            min_temp = fluid.min_temperature
            max_temp = fluid.max_temperature

            sol = root_scalar(residual, method='brentq', bracket=[min_temp, max_temp])
            T = sol.root
            Q = 1.0 if h >= h_vapor else 0.0
        return T, Q


    @staticmethod
    def get_saturation_pressure(
        fluid: Union[str, Dict[str, float]],
        T: float,
        basis: str = "mole"
    ) -> float:

        if isinstance(fluid, str):
            st = pyFluid(pyFluidsList[fluid])

        elif isinstance(fluid, dict):
            fluids = list(fluid.keys())
            fractions = list(fluid.values())

            if len(fluids) == 1:
                st = pyFluid(pyFluidsList[fluids[0]])
            else:
                if basis == "mole":
                    mass_fractions = Fluid.mole_to_mass(fluids, fractions)
                elif basis == "mass":
                    mass_fractions = np.array(fractions, dtype=float)
                else:
                    raise ValueError("basis must be 'mole' or 'mass'!")

                st = pyMixture([pyFluidsList[f] for f in fluids], mass_fractions)

        else:
            raise TypeError("fluid must be a string or dictionary")

        if T < st.min_temperature or T > st.max_temperature:
            raise ValueError(
                f"Temperature {T} K is out of range "
                f"({st.min_temperature} - {st.max_temperature} K)"
            )
        try:
            Psat = st.with_state(pyInput.temperature(T), pyInput.quality(0)).pressure
        except Exception as e:
            raise ValueError(f"Failed to calculate saturation pressure: {e}")

        return Psat


    @staticmethod
    def show_available_fluids():
        for f in pyFluidsList:
            if f.pure and f.name is not None: print(f.name)
        return [f.name for f in pyFluidsList if f.pure and f.name is not None]
    
    @staticmethod
    def get_available_fluids():
        return [f.name for f in pyFluidsList if f.pure and f.name is not None]




if __name__ == "__main__":


    #f = Fluid({"Nitrogen": 0.79, "Oxygen": 0.11, "Methane": 0.1}, basis="mass", P=101325, h=3e7)
    #f = Fluid({"Nitrogen": 1}, P=101325, h=311200)
    #f = Fluid("Methane", P=3e6, Q=0.1)
    f = Fluid("nDodecane", P=101325, T=300)
    print(f)
    #ADD A WAY TO UPDATE THE FLUIDS
    #Fluid.show_available_fluids()
    #print(Fluid.get_saturation_pressure({"Nitrogen": 0.79, "Oxygen": 0.11, "Methane": 0.1}, T=120))


