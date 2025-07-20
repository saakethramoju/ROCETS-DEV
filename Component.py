from typing import Dict, Any, Optional, List
from Port import InputPort, OutputPort
from prettytable import PrettyTable
from Exceptions import (
    PortNotConnectedError, PortTypeError, AmbiguousPortError, NoMatchingPortsError,
    MissingGuessError, MissingGuessKeyError, MissingGuessValueError, GuessResidualMismatchError,
    MissingConfigurationError, MissingConfigurationKeyError, MissingConfigurationValueError,
    InvalidGuessKeyError
)
import re
import numpy as np


class Component:

    required_keys = []
    optional_keys = []

    num_expected_residuals = 1

    def __init__(self, name: str, config: Optional[dict] = None, guess: Optional[dict] = None):
        self.name = name
        self.inputs: Dict[str, InputPort] = {}
        self.outputs: Dict[str, OutputPort] = {}
        self.required_inputs: Dict[str, InputPort] = {}
        self.required_outputs: Dict[str, OutputPort] = {}

        self.configuration: Optional[dict] = config
        self.guess: Optional[dict] = guess


        if config:
            self.set_config(config)

    def validate_all(self, seen_nodes=None):
        self.validate_config()
        self.validate_guess(seen_nodes=seen_nodes)
        self._propagate_iteration_variable_flags()
        self._propagate_guess_variable_flags()
        self.validate_connections_and_iteration_sources()


    # ─────────────────────────────────────────────────────────────────────────────
    # PORT MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────────

    def add_input(self, port_name: str, required: bool = False, iteration_variable: bool = False):
        """Add an input port to the component."""
        port = InputPort(name=port_name, component=self, iteration_variable=iteration_variable)
        self.inputs[port_name] = port
        if required:
            self.required_inputs[port_name] = port
        return port

    def add_output(self, port_name: str, required: bool = False, iteration_variable: bool = False):
        """Add an output port to the component."""
        port = OutputPort(name=port_name, component=self, iteration_variable=iteration_variable)
        self.outputs[port_name] = port
        if required:
            self.required_outputs[port_name] = port
        return port

    def get_iteration_variables(self) -> List[str]:
        return [
            name for name, port in {**self.inputs, **self.outputs}.items()
            if port.iteration_variable
        ]

    def get_guess_variables(self) -> List[str]:
        return [
            name for name, port in {**self.inputs, **self.outputs}.items()
            if port.guess_variable
        ]

    def validate_connections_and_iteration_sources(self):
        """
        Validate:
        1. All required input/output ports are connected.
        2. All iteration variables are either guess variables or connected to a source.
        """
        missing_required = []
        missing_iteration_sources = []

        # Check required port connections
        for name, port in self.required_inputs.items():
            if not port.is_connected():
                missing_required.append(f"Input: {name}")

        for name, port in self.required_outputs.items():
            if not port.is_connected():
                missing_required.append(f"Output: {name}")

        # Check iteration variable sources
        for name, port in {**self.inputs, **self.outputs}.items():
            if port.iteration_variable:
                if not port.guess_variable and not port.is_connected():
                    missing_iteration_sources.append(name)

        # Raise detailed error if any problems found
        error_msgs = []
        if missing_required:
            error_msgs.append(
                "Missing required connections:\n  - " + "\n  - ".join(missing_required)
            )
        if missing_iteration_sources:
            error_msgs.append(
                "Iteration variables without guess or connection:\n  - " + "\n  - ".join(missing_iteration_sources)
            )

        if error_msgs:
            raise PortNotConnectedError(f"{self.name} has unresolved port issues:\n" + "\n\n".join(error_msgs))

        return True


    # ─────────────────────────────────────────────────────────────────────────────
    # CONNECTION LOGIC
    # ─────────────────────────────────────────────────────────────────────────────

    def connect(self, other: "Component", only_required: bool = False):
        """Auto-connect matching ports between components using name normalization."""
        other_inputs = other.required_inputs if only_required else other.inputs
        other_outputs = other.required_outputs if only_required else other.outputs
        connection_made = False

        for self_name, self_port in {**self.inputs, **self.outputs}.items():
            search_ports = other_inputs if isinstance(self_port, OutputPort) else other_outputs
            matches = [
                (name, port) for name, port in search_ports.items()
                if self._normalize(name) == self._normalize(self_name)
            ]
            if len(matches) == 1:
                other_name, other_port = matches[0]
                if isinstance(self_port, OutputPort) and isinstance(other_port, InputPort):
                    self_port.connect(other_port)
                    print(f"[Connected] {self.name}: {self_name} → {other.name}: {other_name}")
                elif isinstance(self_port, InputPort) and isinstance(other_port, OutputPort):
                    other_port.connect(self_port)
                    print(f"[Connected] {other.name}: {other_name} → {self.name}: {self_name}")
                else:
                    raise PortTypeError(f"Invalid connection between {self.name}.{self_name} and {other.name}.{other_name}")
                connection_made = True
            elif len(matches) > 1:
                raise AmbiguousPortError(f"Ambiguous port match for '{self_name}' between {self.name} and {other.name}")

        if not connection_made:
            raise NoMatchingPortsError(f"No ports could be connected between {self.name} and {other.name}.")

    def manual_connect(self, output_name: str, input_comp: "Component", input_name: str):
        """Manually connect one output to another component's input."""
        out = self.outputs.get(output_name)
        inp = input_comp.inputs.get(input_name)
        if out and inp:
            out.connect(inp)
            print(f"[Connected] {self.name}: {output_name} → {input_comp.name}: {input_name}")

    def connect_all_necessary_ports(self, other: "Component"):
        """Connect all required inputs of `other` to matching outputs from this component."""
        self_outputs = {self._normalize(name): port for name, port in self.outputs.items()}
        required_inputs = {
            self._normalize(name): (name, port) for name, port in other.required_inputs.items()
        }

        unmatched = []
        for norm, (orig, inp) in required_inputs.items():
            out = self_outputs.get(norm)
            if out:
                out.connect(inp)
                print(f"[Connected] {self.name}: {out.name} → {other.name}: {orig}")
            else:
                unmatched.append(orig)

        for name in unmatched:
            print(f"[Warning] Could not connect required input '{name}' in {other.name}.")

    # ─────────────────────────────────────────────────────────────────────────────
    # GUESS HANDLING
    # ─────────────────────────────────────────────────────────────────────────────


    def assign_guess_variables(self, priority: Optional[List[str]] = None):
        """
        Assign guess_variable flags to iteration variables, trimming the number to match residuals.
        Priority determines selection order.
        """
        num = self.num_residuals()
        candidates = {
            name: port for name, port in {**self.inputs, **self.outputs}.items()
            if port.iteration_variable
        }

        if len(candidates) < num:
            raise GuessResidualMismatchError(num_guess_vars=len(candidates), num_residuals=num)

        if priority:
            priority_norm = [self._normalize(p) for p in priority]
            sorted_ports = sorted(
                candidates.items(),
                key=lambda item: priority_norm.index(self._normalize(item[0]))
                if self._normalize(item[0]) in priority_norm else float("inf")
            )
        else:
            sorted_ports = list(candidates.items())

        # Reset all first
        for _, port in candidates.items():
            port.guess_variable = False

        # Enable only top N guess variables
        for i, (name, port) in enumerate(sorted_ports):
            if i < num:
                port.guess_variable = True


    def set_guess(self, guess: dict):
        """Set initial guess values for ports using normalized key matching."""
        self.guess = guess
        for key, val in guess.items():
            norm_key = self._normalize(key)
            for name, port in {**self.inputs, **self.outputs}.items():
                if self._normalize(name) == norm_key:
                    self[name] = val
                    break
        return True
    
    def validate_guess(self, seen_nodes: set = None):
        if seen_nodes is None:
            seen_nodes = set()

        if not self.guess:
            self.guess = {}

        required_normalized_keys = {}

        for port in {**self.inputs, **self.outputs}.values():
            shared = port._value
            if not (shared.guess_variable and shared.iteration_variable):
                continue
            if shared in seen_nodes:
                continue
            seen_nodes.add(shared)
            required_normalized_keys[self._normalize(port.name)] = port.name

        if not required_normalized_keys:
            return True

        guess_keys_normalized = {self._normalize(k): k for k in self.guess}
        provided_normalized_keys = set(guess_keys_normalized.keys())

        missing_keys = set(required_normalized_keys.keys()) - provided_normalized_keys
        if missing_keys:
            missing_names = [required_normalized_keys[k] for k in missing_keys]
            raise MissingGuessKeyError(
                f"Missing required initial guess key(s) for {self.name}: {missing_names}"
            )

        extra_keys = provided_normalized_keys - set(required_normalized_keys.keys())
        if extra_keys:
            extra_originals = [guess_keys_normalized[k] for k in extra_keys]
            raise InvalidGuessKeyError(
                f"Unexpected guess key(s) in {self.name}: {extra_originals}. "
                f"Expected only: {[required_normalized_keys[k] for k in required_normalized_keys]}"
            )

        for norm_key in required_normalized_keys:
            original_key = guess_keys_normalized[norm_key]
            if self.guess[original_key] is None:
                raise MissingGuessValueError(f"Initial guess value for '{original_key}' cannot be None")

        return True


    def set_guess_variables(self, names: List[str]):
        """
        Set which iteration_variables should be used as guess variables.

        All other iteration_variables will be deactivated from the guess set.

        Parameters
        ----------
        names : List[str]
            List of port names to be used as guess variables.
            Matching is fuzzy using normalized names.

        Raises
        ------
        InvalidGuessVariableError if any name is not a valid iteration variable.
        """
        from Exceptions import InvalidGuessVariableError

        norm_names = [self._normalize(n) for n in names]

        # Build set of all valid normalized iteration variable names
        valid_vars = {
            self._normalize(name): name
            for name, port in {**self.inputs, **self.outputs}.items()
            if port.iteration_variable
        }

        # Check for any invalid names
        invalid = [name for name in norm_names if name not in valid_vars]
        if invalid:
            raise InvalidGuessVariableError(
                invalid_names=invalid,
                valid_names=list(valid_vars.values())
            )

        # Apply toggles
        for name, port in {**self.inputs, **self.outputs}.items():
            if port.iteration_variable:
                port.guess_variable = self._normalize(name) in norm_names


    def toggle_guess_variable(self, name: str, enable: bool = True):
        """
        Enable or disable a port as an active guess variable.

        Fuzzy-matches port name using normalized comparison.
        The port must already be marked as an iteration variable.

        Parameters:
        ----------
        name : str
            Name of the port to toggle.
        enable : bool, default=True
            True to activate; False to deactivate.

        Raises:
        -------
        ValueError if the port is not an iteration variable.
        KeyError if no matching port is found.
        """
        norm = self._normalize(name)
        for port_name, port in {**self.inputs, **self.outputs}.items():
            if self._normalize(port_name) == norm:
                if not port.iteration_variable:
                    raise ValueError(f"'{port_name}' is not an iteration variable and cannot be toggled.")
                port.guess_variable = enable
                return
        raise KeyError(f"No port found matching '{name}'")


    def get_guess_vector(self):
        """
        Return the current guess vector (only for ports that are both guess and iteration variables).
        """
        return np.array([
            port.value for port in {**self.inputs, **self.outputs}.values()
            if port.guess_variable and port.iteration_variable
        ])

    def set_guess_vector(self, vec: np.ndarray):
        """
        Update the component's values from a new guess vector.
        """
        i = 0
        for port in {**self.inputs, **self.outputs}.values():
            if port.guess_variable and port.iteration_variable:
                port.value = vec[i]
                i += 1

    def _propagate_iteration_variable_flags(self):
        """Ensure all ports in a shared node inherit the iteration_variable flag."""
        seen = set()
        for port in {**self.inputs, **self.outputs}.values():
            shared = port._value
            if shared in seen:
                continue
            seen.add(shared)
            flag = any(p.iteration_variable for p in shared.subscribers)
            for p in shared.subscribers:
                p.iteration_variable = flag

    def _propagate_guess_variable_flags(self):
        """Ensure guess_variable flags are consistent across shared iteration nodes."""
        seen = set()
        for port in {**self.inputs, **self.outputs}.values():
            shared = port._value
            if shared in seen:
                continue
            seen.add(shared)
            if not any(p.iteration_variable for p in shared.subscribers):
                continue
            flag = any(p.guess_variable for p in shared.subscribers)
            for p in shared.subscribers:
                if p.iteration_variable:
                    p.guess_variable = flag

    # ─────────────────────────────────────────────────────────────────────────────
    # NORMALIZATION, ACCESS, RESIDUALS
    # ─────────────────────────────────────────────────────────────────────────────

    def _normalize(self, name: str):
        """Normalize a port name: remove parentheses, lowercase, strip whitespace."""
        return re.sub(r"\(.*?\)", "", name).strip().lower()

    def _resolve_port(self, port_name: str):
        """Resolve a port name to its Input/OutputPort object."""
        norm = self._normalize(port_name)
        for name, port in {**self.inputs, **self.outputs}.items():
            if self._normalize(name) == norm:
                return port
        raise KeyError(f"Port '{port_name}' not found in {self.name}.")

    def __getitem__(self, port_name: str):
        return self._resolve_port(port_name).value

    def __setitem__(self, port_name: str, value: Any):
        self._resolve_port(port_name).value = value

    # ─────────────────────────────────────────────────────────────────────────────
    # CONFIGURATION HANDLING
    # ─────────────────────────────────────────────────────────────────────────────


    def set_config(self, config: dict):
        """Store and normalize config keys, while standardizing string values."""
        self.configuration = config
        self._normalized_config = self._normalize_config_dict(config)

    def _normalize_config_dict(self, config: dict) -> dict:
        """
        Normalize config keys and pre-process values.
        Keys are normalized for lookup.
        String values are trimmed.
        """
        return {
            self._normalize(k): (k, self._clean_config_value(v))
            for k, v in config.items()
        }

    def _clean_config_value(self, v):
        """Clean user-provided value (e.g., trim strings, normalize case where applicable)."""
        if isinstance(v, str):
            cleaned = v.strip()
            # Only lowercase values for known case-sensitive keys
            if "nozzle type" in cleaned.lower() or "combustor area" in cleaned.lower():
                return cleaned.lower()
            return cleaned
        return v

    def _lookup_config_value(self, key: str, condition: bool = True):
        """Lookup value in normalized config, respecting condition."""
        norm = self._normalize(key)
        original, val = self._normalized_config.get(norm, (None, None))
        return val if condition else None
    

    def validate_config(self):
        """Validate required config keys are present and non-null."""
        required_keys = getattr(self.__class__, "required_keys", [])

        # If no required keys, skip config validation
        if not required_keys:
            return

        if not self.configuration:
            raise MissingConfigurationError(f"No configuration provided for {self.name}")

        for key in required_keys:
            norm = self._normalize(key)
            if norm not in self._normalized_config:
                raise MissingConfigurationKeyError(f"Missing required key: '{key}'")
            original_key, value = self._normalized_config[norm]
            if value is None:
                raise MissingConfigurationValueError(f"Key '{original_key}' is present but has value None")

    # ─────────────────────────────────────────────────────────────────────────────
    # FORMATTED REPRESENTATION
    # ─────────────────────────────────────────────────────────────────────────────
    def __repr__(self):

        def build_table(port_dict, direction_label, required_set):
            table = PrettyTable()
            table.title = f"{direction_label} Ports"
            table.field_names = [
                "Port Name", "Connections", "Required", "Iteration Variable", "Guess Variable", "Current Value"
            ]
            table.align["Port Name"] = "l"  # Left-justify port names

            for name, port in port_dict.items():
                conns = ", ".join(f"{p.name} in '{p.component.name}'" for p in port.connected_ports) if port.is_connected() else "None"
                required = "Yes" if name in required_set else "No"
                iteration = "Yes" if port.iteration_variable else "No"
                guess = "Yes" if port.guess_variable else "No"
                value = port.value if port.value is not None else "—"
                table.add_row([name, conns, required, iteration, guess, value])

            return table

        input_table = build_table(self.inputs, "Input", self.required_inputs)
        output_table = build_table(self.outputs, "Output", self.required_outputs)

        return f"\n========== Component: {self.name} ==========\n{input_table}\n{output_table}"
        
    def print_iteration_variable_table(self):
        """
        Pretty-print a table of all iteration variables, indicating which are guess variables.
        """

        table = PrettyTable()
        table.field_names = ["Port Name", "Direction", "Guess Variable", "Current Value"]
        table.align["Port Name"] = "l"  # Left-justify port names

        for name, port in {**self.inputs, **self.outputs}.items():
            if port.iteration_variable:
                direction = "Input" if name in self.inputs else "Output"
                is_guess = "Yes" if port.guess_variable else "-"
                value = port.value if port.value is not None else "—"
                table.add_row([name, direction, is_guess, value])

        print(f"\n========== Iteration Variables for Component: {self.name} ==========")
        print(table)

    def print_guess_variable_table(self):
        """
        Pretty-print a table of all guess variables, indicating whether they're iteration variables.
        """

        table = PrettyTable()
        table.field_names = ["Port Name", "Direction", "Iteration Variable", "Current Value"]
        table.align["Port Name"] = "l"  # Left-justify port names

        for name, port in {**self.inputs, **self.outputs}.items():
            if port.guess_variable:
                direction = "Input" if name in self.inputs else "Output"
                is_iter = "Yes" if port.iteration_variable else "-"
                value = port.value if port.value is not None else "—"
                table.add_row([name, direction, is_iter, value])

        print(f"\n========== Guess Variables for Component: {self.name} ==========")
        print(table)

    # ─────────────────────────────────────────────────────────────────────────────
    # RESIDUALS / SOLVER SUPPORT
    # ─────────────────────────────────────────────────────────────────────────────

    def residuals(self) -> List[float]:
        """
        Define the residuals for this component.
        Subclasses should override this to return a list of residual expressions.
        """
        return []  # default placeholder residual

    def num_residuals(self) -> int:
        """
        Return the number of residual equations for this component.
        Used to determine how many guess variables should be selected.
        """
        self._num_residuals = len(self.residuals())
        return self._num_residuals

    def on_steady_state_solve(self, solution):
        """Optional hook to run after steady-state solver succeeds."""
        pass

if __name__ == "__main__":

    '''
    # Create two components
    injector = Component("Injector")
    chamber = Component("Chamber")

    # Define ports
    injector.add_output("Chamber Pressure (psia)")
    chamber.add_input("Chamber Pressure (psia)", required=True)

    # Option 1: Set value on OUTPUT before connecting
    injector["Chamber Pressure (psia)"] = 500
    injector.connect(chamber)
    print("\nAfter setting injector's output value and connecting:")
    print(f"Injector port value: {injector['Chamber Pressure (psia)']}")
    print(f"Chamber port value: {chamber['Chamber Pressure (psia)']}")

    # Option 2: Reset, set input value first
    print("\nNow reversing the value direction...")

    # Reset ports manually for clean test (simulate new components)
    injector = Component("Injector")
    chamber = Component("Chamber")
    injector.add_output("Chamber Pressure (psia)")
    chamber.add_input("Chamber Pressure (psia)", required=True)

    chamber["Chamber Pressure (psia)"] = 600
    chamber.connect(injector)  # This time connecting from input side
    print("\nAfter setting chamber's input value and connecting:")
    print(f"Injector port value: {injector['Chamber Pressure (psia)']}")
    print(f"Chamber port value: {chamber['Chamber Pressure (psia)']}")

    # Option 3: Both sides set
    print("\nNow testing override...")
    injector = Component("Injector")
    chamber = Component("Chamber")
    injector.add_output("Chamber Pressure (psia)")
    chamber.add_input("Chamber Pressure (psia)", required=True)

    injector["Chamber Pressure (psia)"] = 700
    chamber["Chamber Pressure (psia)"] = 800
    injector.connect(chamber)
    print("\nAfter both sides had values and connecting (override expected):")
    print(f"Injector port value: {injector['Chamber Pressure (psia)']}")
    print(f"Chamber port value: {chamber['Chamber Pressure (psia)']} \n")'''



    print("Testing shared value bus")

    injector = Component("Injector")
    chamber = Component("Chamber")
    sensor = Component("Sensor")

    injector.add_output("Flow (kg/s)")
    chamber.add_input("Flow", required=True)
    sensor.add_input("Flow", required=True)

    print(injector)
    print(chamber)
    print(sensor)

    print(chamber['Flow'])

    injector["Flow"] = 100
    injector.connect(chamber)
    sensor.connect(injector)

    print(chamber["flow"])
    print(sensor['flow'])

    sensor["flow"] = 25

    print(chamber["Flow"])
    print(injector['flow'])




