import argparse
import numpy as np
from qiskit import QuantumCircuit
from utils.circuit_loader import load_circuit_from_file_or_string, circuit_to_target_state
from utils.quantum_metrics import (
    embed_1q_gate,
    cnot_matrix,
    state_to_density_matrix,
    state_fidelity,
    purity,
    H_GATE,
    X_GATE,
    rx,
    ry,
    rz,
)
from envs.noise_models import NoiseModel


def evaluate_custom_circuit_under_noise(
    qc: QuantumCircuit, p_cnot: float = 0.04
) -> dict:
    """
    Executes a user-supplied Qiskit QuantumCircuit through the open-system
    density matrix simulator with CNOT depolarizing noise.
    """
    num_qubits = qc.num_qubits
    dim = 2**num_qubits

    # Ideal target pure state (unitary evolution with no noise)
    ideal_psi = circuit_to_target_state(qc)
    ideal_rho = state_to_density_matrix(ideal_psi)

    # Open system density matrix starting from |0...0>
    init_psi = np.zeros(dim, dtype=np.complex128)
    init_psi[0] = 1.0
    rho = state_to_density_matrix(init_psi)

    noise_model = NoiseModel(p_cnot=p_cnot)
    cnot_count = 0
    gate_history = []

    # Map Qiskit gate operations
    for instruction in qc.data:
        op = instruction.operation
        name = op.name.lower()
        q_indices = [qc.find_bit(q).index for q in instruction.qubits]

        if name == "h":
            u = embed_1q_gate(H_GATE, q_indices[0], num_qubits)
            rho = u @ rho @ u.conj().T
            gate_history.append(f"H(q{q_indices[0]})")
        elif name == "x":
            u = embed_1q_gate(X_GATE, q_indices[0], num_qubits)
            rho = u @ rho @ u.conj().T
            gate_history.append(f"X(q{q_indices[0]})")
        elif name in ["ry", "rx", "rz"]:
            theta = float(op.params[0])
            mat = ry(theta) if name == "ry" else (rx(theta) if name == "rx" else rz(theta))
            u = embed_1q_gate(mat, q_indices[0], num_qubits)
            rho = u @ rho @ u.conj().T
            gate_history.append(f"{name.upper()}({theta:.2f}, q{q_indices[0]})")
        elif name in ["cx", "cnot"]:
            c, t = q_indices[0], q_indices[1]
            u = cnot_matrix(c, t, num_qubits)
            rho = u @ rho @ u.conj().T
            cnot_count += 1
            gate_history.append(f"CNOT(c{c}->t{t})")
            # Apply 2-qubit depolarizing noise
            rho = noise_model.apply_cnot_depolarizing(rho, c, t, num_qubits)
        elif name in ["barrier", "measure"]:
            continue
        else:
            print(f"[Warning] Gate '{name}' not in native dictionary, skipping simulation.")

    final_fidelity = state_fidelity(rho, ideal_psi)
    final_purity = purity(rho)

    return {
        "num_qubits": num_qubits,
        "total_gates": len(gate_history),
        "cnot_count": cnot_count,
        "final_fidelity": final_fidelity,
        "final_purity": final_purity,
        "gate_history": gate_history,
        "circuit_diagram": qc.draw(output="text"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate any Qiskit circuit under CNOT noise.")
    parser.add_argument(
        "--circuit",
        type=str,
        default="redundant_bell",
        help="Path to .qasm file or built-in template ('bell', 'redundant_bell', 'ghz3', 'w3')",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.04,
        help="CNOT depolarizing error rate p",
    )
    args = parser.parse_args()

    qc = load_circuit_from_file_or_string(args.circuit)

    print("=" * 65)
    print("USER INPUT CIRCUIT BENCHMARK UNDER OPEN-SYSTEM NOISE")
    print(f"Circuit Source : {args.circuit}")
    print(f"CNOT Noise     : p = {args.noise:.4f}")
    print("=" * 65)
    print("Circuit Diagram:")
    print(qc.draw(output="text"))
    print("-" * 65)

    res = evaluate_custom_circuit_under_noise(qc, p_cnot=args.noise)
    print(f"Final State Fidelity : {res['final_fidelity']:.4f}")
    print(f"Final State Purity   : {res['final_purity']:.4f}")
    print(f"Total CNOTs Applied  : {res['cnot_count']}")
    print(f"Total Gates          : {res['total_gates']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
