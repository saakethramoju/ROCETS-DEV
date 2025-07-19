from Component import Component
from typing import Optional
from Exceptions import (PortNotConnectedError, MissingConfigurationError, MissingConfigurationKeyError,
                        MissingConfigurationValueError)

class Injector(Component):
    def __init__(self, name: str, config: Optional[dict] = None):
        super().__init__(name)
        self.configuration = config
 