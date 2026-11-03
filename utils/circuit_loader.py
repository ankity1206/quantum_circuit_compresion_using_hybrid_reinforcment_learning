import os
import re
from typing import Optional, Union, List, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix


def load_circuit_from_file_or_string(circuit_input: str) -> QuantumCircuit:
    """
    Loads a Qiskit QuantumCircuit from:
    1. A .qasm file path
    2. A raw OpenQASM string
    3. A pre-defined template name ('bell', 'ghz3', 'w3', 'redundant_bell')
    """
    if os.path.isfile(circuit_input):
        return QuantumCircuit.from_qasm_file(circuit_input)

    # Built-in circuit templates
    if circuit_input.lower() == "bell":
        qc = QuantumCircuit(2, name="Bell_Target")
        qc.h(0)
        qc.cx(0, 1)
        return qc
    elif circuit_input.lower() == "ghz3":
        qc = QuantumCircuit(3, name="GHZ3_Target")
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        return qc
    elif circuit_input.lower() == "w3":
        qc = QuantumCircuit(3, name="W3_Target")
        qc.ry(2 * np.arccos(1 / np.sqrt(3)), 0)
        qc.ch(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.x(0)
        return qc
    elif circuit_input.lower() == "redundant_bell":
        qc = QuantumCircuit(2, name="Redundant_Bell_Target")
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)  # Redundant pair
        qc.cx(0, 1)
        return qc
    elif circuit_input.lower() in ["redundant_teleportation", "teleportation"]:
        qc = QuantumCircuit(3, name="Redundant_Teleportation")
        qc.h(0)
        qc.h(1)
        qc.cx(1, 2)
        # Redundant CNOT pair that cancels out (CX * CX = I)
        qc.cx(1, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        return qc

    # Attempt parsing as raw QASM string
    if "OPENQASM" in circuit_input:
        return QuantumCircuit.from_qasm_str(circuit_input)

    raise ValueError(
        f"Could not parse circuit input: '{circuit_input}'. "
        f"Provide a path to a .qasm file or one of: ['bell', 'ghz3', 'w3', 'redundant_bell']"
    )


def circuit_to_target_state(qc: QuantumCircuit) -> np.ndarray:
    """Extracts statevector from a Qiskit QuantumCircuit, matching convention."""
    sv = Statevector.from_instruction(qc).data
    return np.asarray(sv, dtype=np.complex128)


def history_to_qiskit_circuit(history: List[str], num_qubits: int = 2) -> QuantumCircuit:
    """
    Converts an agent's synthesized gate action history into a Qiskit QuantumCircuit.
    Example history: ['H(q0)', 'CNOT(c0->t1)', 'STOP']
    """
    qc = QuantumCircuit(num_qubits, name="Synthesized_Output_Circuit")

    for act in history:
        act = act.strip()
        if act == "STOP":
            continue

        # Match CNOT: CNOT(c0->t1)
        m_cnot = re.match(r"CNOT\(c(\d+)->t(\d+)\)", act)
        if m_cnot:
            c, t = int(m_cnot.group(1)), int(m_cnot.group(2))
            qc.cx(c, t)
            continue

        # Match single qubit gates: H(q0), X(q1), etc.
        m_1q = re.match(r"([A-Za-z0-9_]+)\(q(\d+)\)", act)
        if m_1q:
            gate_name, q = m_1q.group(1), int(m_1q.group(2))
            if gate_name == "H":
                qc.h(q)
            elif gate_name == "X":
                qc.x(q)
            elif gate_name == "Rx_pi2":
                qc.rx(np.pi / 2, q)
            elif gate_name == "Rx_neg_pi2":
                qc.rx(-np.pi / 2, q)
            elif gate_name == "Ry_pi2":
                qc.ry(np.pi / 2, q)
            elif gate_name == "Ry_neg_pi2":
                qc.ry(-np.pi / 2, q)
            elif gate_name == "Rz_pi2":
                qc.rz(np.pi / 2, q)
            elif gate_name == "Rz_neg_pi2":
                qc.rz(-np.pi / 2, q)
            else:
                pass

    return qc
