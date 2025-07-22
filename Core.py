import os
import re
import yaml
from pathlib import Path
from typing import Any, List, Optional
from prettytable import PrettyTable
from difflib import get_close_matches

from Exceptions import (MissingConfigurationError, MissingConfigurationKeyError, MissingConfigurationValueError,
                        MissingPortError, PortTypeError, MissingMassConservationEquation, MissingFlowPortError,
                        FlowPortTypeError)


class Component:

    required_config_keys = []

    optional_config_keys = []

    config_keys = required_config_keys + optional_config_keys

    system_variables: List[str] = []

    def __init__(self, name: str):
        self.name = name
        self.ports: List["Port"] = []
        self.system = None 

    def add_inflow(self, name: str) -> "InFlow":
        port = InFlow(name, self)
        self.ports.append(port)
        return port

    def add_outflow(self, name: str) -> "OutFlow":
        port = OutFlow(name, self)
        self.ports.append(port)
        return port


    def add_input(self, name: str) -> "InputPort":
        port = InputPort(name, self)
        self.ports.append(port)

        # Initialize instance-level system_variables if needed
        if not hasattr(self, "system_variables"):
            self.system_variables = []

        if name not in self.system_variables:
            self.system_variables.append(name)

        return port

    def add_output(self, name: str) -> "OutputPort":
        port = OutputPort(name, self)
        self.ports.append(port)
        return port

    def _normalize(self, name: str):
        return re.sub(r"\(.*?\)", "", name).strip().lower()
    
    def connect(self, other: "Component"):
        """Auto-connect matching ports between this component and another."""
        # Build normalized port name lookup
        self_outputs = {
            self._normalize(p.name): p for p in self.ports if isinstance(p, OutputPort)
        }
        self_inputs = {
            self._normalize(p.name): p for p in self.ports if isinstance(p, InputPort)
        }

        other_outputs = {
            other._normalize(p.name): p for p in other.ports if isinstance(p, OutputPort)
        }
        other_inputs = {
            other._normalize(p.name): p for p in other.ports if isinstance(p, InputPort)
        }

        # Try to match self.outputs → other.inputs
        for name, out_port in self_outputs.items():
            if name in other_inputs:
                in_port = other_inputs[name]
                out_port.connect(in_port)

        # Try to match other.outputs → self.inputs
        for name, out_port in other_outputs.items():
            if name in self_inputs:
                in_port = self_inputs[name]
                out_port.connect(in_port)

        if self.system and other.system is None:
            self.system.add_component(other)
        elif other.system and self.system is None:
            other.system.add_component(self)


    def connect_flow(self, port_name_self: str, other: "Component", port_name_other: str):
        """Connect a flow port from self to a flow port on another component.

        Requires explicit port names. Automatically handles InFlow ↔ OutFlow logic.

        Raises:
            MissingFlowPortError: If one or both ports cannot be found.
            FlowPortTypeError: If both ports are the same type (InFlow ↔ InFlow or OutFlow ↔ OutFlow).
        """
        # Normalize input
        norm_self = self._normalize(port_name_self)
        norm_other = other._normalize(port_name_other)

        # Find ports
        port_self = next((p for p in self.ports if self._normalize(p.name) == norm_self), None)
        port_other = next((p for p in other.ports if other._normalize(p.name) == norm_other), None)

        # Check both ports exist
        if not port_self or not port_other:
            raise MissingFlowPortError(
                f"Could not find flow ports: '{port_name_self}' on '{self.name}' "
                f"or '{port_name_other}' on '{other.name}'."
            )

        # Check types
        valid_pair = (
            (isinstance(port_self, InFlow) and isinstance(port_other, OutFlow)) or
            (isinstance(port_self, OutFlow) and isinstance(port_other, InFlow))
        )

        if not valid_pair:
            raise FlowPortTypeError(
                f"Invalid connection: '{port_name_self}' on '{self.name}' ({type(port_self).__name__}) "
                f"↔ '{port_name_other}' on '{other.name}' ({type(port_other).__name__})."
            )

        # Connect them
        port_self.connect(port_other)


    def set_config(self, config: dict):
        """Store config with fuzzy-matched canonical keys (both required and optional)."""
        self.configuration = {}

        for input_key, value in config.items():
            norm_input = self._normalize(input_key)
            match = self._fuzzy_match_config_key(norm_input, search_keys=self.config_keys)
            if match:
                self.configuration[match] = value
            else:
                print(f"Warning: Unrecognized config key '{input_key}' for component '{self.name}'")

        self._normalized_config = {
            self._normalize(k): v for k, v in self.configuration.items()
        }

    def validate_config(self, return_missing: bool = False):
        """Ensure config exists, required keys are present, and values are valid."""
        
        if not hasattr(self, "configuration") or not isinstance(self.configuration, dict):
            raise MissingConfigurationError(
                f"Component '{self.name}' has no configuration set. Call set_config(config_dict) first."
            )

        missing_keys = []
        missing_values = []

        for key in self.required_config_keys:
            if key not in self.configuration:
                missing_keys.append(key)
            elif self.configuration[key] in [None, "", "—"]:
                missing_values.append(key)

        if missing_keys:
            raise MissingConfigurationKeyError(
                f"Component '{self.name}' is missing required config keys: {', '.join(missing_keys)}"
            )

        if missing_values:
            raise MissingConfigurationValueError(
                f"Component '{self.name}' has required config keys with missing/invalid values: "
                f"{', '.join(missing_values)}"
            )

        if return_missing:
            return {"missing_keys": missing_keys, "missing_values": missing_values}


    def config_summary(self):
        """Pretty print the resolved configuration, including optional keys."""
        config = getattr(self, "configuration", {})

        table = PrettyTable()
        table.title = f"Configuration Summary for {self.name}"
        table.field_names = ["Config Key", "Value", "Required?"]

        for key in self.required_config_keys + self.optional_config_keys:
            value = config.get(key, "—")
            is_required = "Yes" if key in self.required_config_keys else "No"
            table.add_row([key, value, is_required])

        #print(str(table))
        return str(table)
    

    def _normalize(self, name: str) -> str:
        return re.sub(r"\(.*?\)", "", name).strip().lower()

    def _fuzzy_match_config_key(self, normalized_input: str, search_keys=None) -> str:
        """Match normalized input to config keys (required + optional)."""
        keys_to_search = search_keys or self.config_keys
        normalized_keys = {self._normalize(k): k for k in keys_to_search}
        match = get_close_matches(normalized_input, normalized_keys.keys(), n=1, cutoff=0.8)
        return normalized_keys[match[0]] if match else None


    def generate_configuration_template(self, file_name: str = None):
        """
        Write a YAML configuration template for this component, listing required and optional keys.
        If a file with the same name already exists, a numbered version will be written.
        """
        config_dict = {}

        for key in getattr(self, "required_config_keys", []):
            config_dict[key] = None

        for key in getattr(self, "optional_config_keys", []):
            config_dict[key] = None

        if file_name is None:
            file_name = f"{self.name} Configuration.yaml"

        # Ensure file name is unique if it exists
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(file_name):
            file_name = f"{base}_{counter}{ext}"
            counter += 1

        # Dump YAML to string for cleanup
        yaml_string = yaml.dump(
            config_dict,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )

        # Replace ": null" with just ":" for clarity
        cleaned_yaml = yaml_string.replace(": null", ":")

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"# Configuration Template for {self.name}\n")
            f.write("# Fill in the required values. Optional values may be left blank.\n\n")
            f.write(cleaned_yaml)

        print(f"[✓] Configuration template written to: {os.path.abspath(file_name)}")


    def read_configuration(self, file_path: str):
        """
        Read a YAML configuration file and apply the configuration to this component.
        
        Parameters:
            file_path (str): Path to the YAML configuration file.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
            yaml.YAMLError: If the file is not a valid YAML file.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                config_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file '{file_path}': {e}")

        if not isinstance(config_data, dict):
            raise ValueError(f"YAML file '{file_path}' must contain a dictionary at the top level.")

        self.set_config(config_data)
        print(f"[✓] Configuration successfully loaded from '{file_path}'")

    def add_system_target(self, key: str, value: Any):
        if not hasattr(self, "system_targets"):
            self.system_targets = []
        if not hasattr(self, "_system_values"):
            self._system_values = {}

        if key not in self.system_targets:
            self.system_targets.append(key)
        self._system_values[key] = value

    def __getitem__(self, key: str):
        norm_key = self._normalize(key)

        # 1. Flow Ports (mass flow access)
        for port in self.ports:
            if isinstance(port, FlowPort) and self._normalize(port.name) == norm_key:
                return port.get_mass_flow()

        # 2. Regular Ports
        for port in self.ports:
            if self._normalize(port.name) == norm_key:
                return port.get_value()

        # 3. "geometry"
        if norm_key == "geometry":
            try:
                self.validate_config()
                return self.generate_geometry()
            except Exception:
                return None

        # 4. Config keys
        for config_key in self.required_config_keys + self.optional_config_keys:
            if self._normalize(config_key) == norm_key:
                return self.configuration.get(config_key)

        # 5. System targets
        if hasattr(self, "system_targets") and hasattr(self, "_system_values"):
            for sys_key in self.system_targets:
                if self._normalize(sys_key) == norm_key:
                    return self._system_values.get(sys_key, None)

        raise MissingPortError(
            f"No port or configuration/system key matching '{key}' found in component '{self.name}'"
        )

    def __setitem__(self, key: str, value):
        norm_key = self._normalize(key)

        # 1. Flow Ports (mass flow setter)
        for port in self.ports:
            if isinstance(port, FlowPort) and self._normalize(port.name) == norm_key:
                port.set_mass_flow(value)
                return

        # 2. Regular Ports
        for port in self.ports:
            if self._normalize(port.name) == norm_key:
                port.set_value(value)
                return

        # 3. Config keys
        match = self._fuzzy_match_config_key(norm_key, search_keys=self.config_keys)
        if match:
            if not hasattr(self, "configuration"):
                self.configuration = {}
            self.configuration[match] = value
            return

        # 4. System targets
        if hasattr(self, "system_targets"):
            for target_key in self.system_targets:
                if self._normalize(target_key) == norm_key:
                    if not hasattr(self, "_system_values"):
                        self._system_values = {}
                    self._system_values[target_key] = value
                    return

        raise MissingPortError(
            f"No port or configuration/system key matching '{key}' found in component '{self.name}'"
        )

    def __repr__(self):
        def build_table(ports, direction_label):
            table = PrettyTable()
            table.title = f"{direction_label} Ports for {self.name}"
            table.field_names = ["Port", "Connected To", "Node", "Current Value"]
            table.align["Port"] = "l"

            for port in ports:
                port_id = port.name

                if isinstance(port, FlowPort):
                    node_name = port.flow_node.name if port.flow_node else "—"
                    value = port.get_mass_flow() if port.get_mass_flow() is not None else "—"
                    connected_ports = [
                        f"{p.name} [{p.owner.name}]"
                        for p in port.flow_node.ports
                        if p is not port
                    ] if port.flow_node else []

                else:
                    node_name = port.node.name if port.node else "—"
                    value = port.get_value() if port.get_value() is not None else "—"
                    connected_ports = [
                        f"{p.name} [{p.owner.name}]"
                        for p in port.node.ports
                        if p is not port
                    ] if port.node else []

                if connected_ports:
                    table.add_row([port_id, connected_ports[0], node_name, value])
                    for conn in connected_ports[1:]:
                        table.add_row(["", conn, "", ""])
                else:
                    table.add_row([port_id, "—", node_name, value])

            return table

        input_ports = [p for p in self.ports if isinstance(p, InputPort)]
        output_ports = [p for p in self.ports if isinstance(p, OutputPort)]
        inflows = [p for p in self.ports if isinstance(p, InFlow)]
        outflows = [p for p in self.ports if isinstance(p, OutFlow)]

        input_table = build_table(input_ports, "Input")
        output_table = build_table(output_ports, "Output")
        inflow_table = build_table(inflows, "Inflow")
        outflow_table = build_table(outflows, "Outflow")

        return (
            f"\n========== Component: {self.name} ==========\n"
            f"{input_table}\n"
            f"{output_table}\n"
            f"{inflow_table}\n"
            f"{outflow_table}"
        )

    def mass_flow(self):
        raise MissingMassConservationEquation("Make sure that the component has an implmented mass conservation equation!")





class Node:
    def __init__(self, name: Optional[str] = None):
        self.name = name or f"Node_{id(self)}"
        self.ports: List["Port"] = []
        self._value: Any = None

    def connect(self, port: "Port"):
        if port not in self.ports:
            self.ports.append(port)
            port.node = self

    def set_value(self, value: Any):
        self._value = value
        for port in self.ports:
            port._local_value = value  # Update each connected port's cached value

    def get_value(self) -> Any:
        # Try to retrieve from node value first
        if self._value is not None:
            return self._value
        # If not set, look at individual ports
        for port in self.ports:
            if port._local_value is not None:
                self.set_value(port._local_value)
                return self._value
        return None

    def __repr__(self):
        return f"<Node name={self.name} value={self._value} ports={len(self.ports)}>"
    


class FlowNode:
    def __init__(self, name=None):
        self.name = name or f"FlowNode_{id(self)}"
        self.ports = []  # Only FlowPorts
        self.residual = 0.0  # To store mass conservation residual

    def connect(self, port: "FlowPort"):
        if port not in self.ports:
            self.ports.append(port)
            port.flow_node = self

    def compute_residual(self):
        """Sum all inflows (positive) and outflows (negative)"""
        residual = 0.0
        for port in self.ports:
            flow = port.get_mass_flow() or 0.0
            residual += flow if isinstance(port, InFlow) else -flow
        self.residual = residual
        return residual

    def __repr__(self):
        return f"<FlowNode {self.name} residual={self.residual:.4f}>"


class Port:
    def __init__(self, name: str, owner: Component):
        self.name = name
        self.owner = owner
        self.node: Optional[Node] = None
        self._local_value: Any = None  # Port-specific cached value

    def connect(self, other: "Port"):
        if type(self) == type(other):
            raise PortTypeError(f"Cannot connect two ports of the same type: {self} ↔ {other}")

        # Always merge under a single shared node
        if self.node and other.node:
            if self.node is not other.node:
                for port in other.node.ports[:]:
                    self.node.connect(port)
        elif self.node:
            self.node.connect(other)
        elif other.node:
            other.node.connect(self)
        else:
            new_node = Node()
            new_node.connect(self)
            new_node.connect(other)

    def get_value(self):
        if self.node:
            return self.node.get_value()
        return self._local_value

    def set_value(self, value: Any):
        self._local_value = value
        if self.node:
            self.node.set_value(value)

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} owner={self.owner.name}>"

    def __str__(self):
        return f"{self.owner.name}.{self.name}"


class InputPort(Port):
    pass


class OutputPort(Port):
    pass

class FlowPort(Port):
    def __init__(self, name: str, owner: Component):
        super().__init__(name, owner)  # Initialize base Port attributes (like node)
        self.flow_node: Optional[FlowNode] = None
        self._mass_flow: Optional[float] = None  # in lbm/s or appropriate unit

    def connect(self, other: "FlowPort"):
        if type(self) == type(other):
            raise PortTypeError("Cannot connect two ports of the same flow direction.")
        if self.flow_node and other.flow_node:
            if self.flow_node is not other.flow_node:
                for port in other.flow_node.ports[:]:
                    self.flow_node.connect(port)
        elif self.flow_node:
            self.flow_node.connect(other)
        elif other.flow_node:
            other.flow_node.connect(self)
        else:
            new_node = FlowNode()
            new_node.connect(self)
            new_node.connect(other)

    def get_mass_flow(self) -> Optional[float]:
        return self._mass_flow

    def set_mass_flow(self, value: float):
        self._mass_flow = value
        if self.flow_node:
            self.flow_node.compute_residual()

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name} of {self.owner.name}>"

class InFlow(FlowPort):
    pass

class OutFlow(FlowPort):
    pass

