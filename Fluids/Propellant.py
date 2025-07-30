# Propellant.py
from rocketprops.rocket_prop import get_prop
from .BaseFluid import BaseFluid

class Propellant(BaseFluid):
    def __init__(self, name: str, *, T=None, P=None):
        super().__init__(name)
        self._prop = get_prop(name)
        self._T = T
        self._P = P
        self._X = None

    @property
    def T(self): return self._T
    @T.setter
    def T(self, value): self._T = value

    @property
    def P(self): return self._P
    @P.setter
    def P(self, value): self._P = value

    @property
    def X(self): return self._X
    @X.setter
    def X(self, value): self._X = value  # RP may not support this yet

    def set_state(self, *, T=None, P=None, X=None):
        self._T = T
        self._P = P
        self._X = X

    @property
    def density(self):
        sg = self._prop.specific_gravity()
        return sg * 1000  # Assuming water = 1000 kg/m³

    @property
    def viscosity(self):
        return self._prop.viscosity(self._T)

    @property
    def cp(self):
        return self._prop.cp(self._T)

    @property
    def enthalpy(self):
        return self._prop.hv(self._T)

    @property
    def thermal_conductivity(self):
        return self._prop.k(self._T)

    @property
    def prandtl(self):
        return self._prop.prandtl(self._T)

    @property
    def speed_of_sound(self):
        return self._prop.speed_of_sound(self._T)

    @property
    def phase(self):
        if self._T and self._T > self._prop.Tb():
            return "Gas"
        return "Liquid"

    @property
    def molecular_weight(self):
        return self._prop.M()

    def __str__(self):
        return f"Propellant: {self.name}, T={self.T}, P={self.P}"
