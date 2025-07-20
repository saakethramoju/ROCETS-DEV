class MissingConfigurationError(Exception):
    pass

class PortNotConnectedError(Exception):
    pass

class MissingIterationSourceError(Exception):
    pass

class NoMatchingPortsError(Exception):
    pass

class PortTypeError(Exception):
    pass

class AmbiguousPortError(Exception):
    pass

class MissingConfigurationKeyError(KeyError):
    pass

class MissingConfigurationValueError(ValueError):
    pass

class MissingGuessError(Exception):
    pass

class MissingGuessKeyError(KeyError):
    pass


class MissingGuessValueError(ValueError):
    pass

class GuessResidualMismatchError(Exception):
    def __init__(self, num_guess_vars, num_residuals):
        super().__init__(
            f"Mismatch between guess variables and residuals: "
            f"{num_guess_vars} guess variable(s) vs {num_residuals} residual(s). "
            f"These must match for the solver to work."
        )
        self.num_guess_vars = num_guess_vars
        self.num_residuals = num_residuals

class SteadyStateSolveError(Exception):
    def __init__(self, component_name: str, message: str, guess_vars: list):
        full_msg = (
            f"Solver failed for component '{component_name}': {message}\n"
            f"Active guess variables: {guess_vars}\n"
            f"Suggestion: consider modifying the initial guess or choosing different guess variables."
        )
        super().__init__(full_msg)
        self.component_name = component_name
        self.guess_vars = guess_vars


class InvalidGuessVariableError(Exception):
    def __init__(self, invalid_names: list, valid_names: list):
        self.invalid_names = invalid_names
        self.valid_names = valid_names

        message = (
            f"Invalid guess variable(s): {', '.join(invalid_names)}\n"
            f"Valid iteration variables are: {', '.join(valid_names)}"
        )
        super().__init__(message)

class MissingMixtureRatioError(Exception):
    pass

class InvalidGuessKeyError(Exception):
    def __init__(self, component_name: str, provided_keys: list, expected_keys: list):
        unexpected = [k for k in provided_keys if k not in expected_keys]
        missing = [k for k in expected_keys if k not in provided_keys]

        message = f"Invalid guess keys for {component_name}."
        if unexpected:
            message += f" Unexpected keys provided: {unexpected}."
        if missing:
            message += f" Missing required guess keys: {missing}."

        super().__init__(message)
        self.component_name = component_name
        self.provided_keys = provided_keys
        self.expected_keys = expected_keys
        self.unexpected_keys = unexpected
        self.missing_keys = missing
