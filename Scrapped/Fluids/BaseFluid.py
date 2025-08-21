from abc import ABC, abstractmethod
from typing import Optional

class BaseFluid(ABC):
    def __init__(self, name: str):
        self.name = name

    # --- Required thermodynamic input interface ---
    @property
    @abstractmethod
    def T(self) -> Optional[float]: ...
    @T.setter
    @abstractmethod
    def T(self, value: float): ...

    @property
    @abstractmethod
    def P(self) -> Optional[float]: ...
    @P.setter
    @abstractmethod
    def P(self, value: float): ...

    @property
    @abstractmethod
    def X(self) -> Optional[float]: ...
    @X.setter
    @abstractmethod
    def X(self, value: float): ...

    @abstractmethod
    def set_state(self, *, T=None, P=None, X=None): ...

    # --- Required fluid properties used in Fluid/Mixture ---
    @property
    @abstractmethod
    def density(self) -> float: ...

    @property
    @abstractmethod
    def viscosity(self) -> float: ...

    @property
    @abstractmethod
    def cp(self) -> float: ...

    @property
    @abstractmethod
    def thermal_conductivity(self) -> float: ...

    @property
    @abstractmethod
    def speed_of_sound(self) -> float: ...

    @property
    @abstractmethod
    def enthalpy(self) -> float: ...

    @property
    @abstractmethod
    def prandtl(self) -> float: ...

    @property
    @abstractmethod
    def molecular_weight(self) -> float: ...

    @property
    @abstractmethod
    def critical_temperature(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def critical_pressure(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def min_temperature(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def max_temperature(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def min_pressure(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def max_pressure(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def phase(self) -> str: ...

    # --- Optional: for mixtures only ---
    @property
    def mole_fractions(self) -> dict[str, float]:
        raise NotImplementedError

    @mole_fractions.setter
    def mole_fractions(self, value: dict[str, float]):
        raise NotImplementedError

    @property
    def mass_fractions(self) -> dict[str, float]:
        raise NotImplementedError

    @mass_fractions.setter
    def mass_fractions(self, value: dict[str, float]):
        raise NotImplementedError
