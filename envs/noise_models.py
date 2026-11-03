import itertools
import numpy as np
from utils.quantum_metrics import PAULI_DICT, I2, X_GATE, Y_GATE, Z_GATE


class NoiseModel:
    """
    Open quantum system noise model using operator-sum (Kraus/Pauli) representation.
    Supports single-qubit gate depolarizing noise and two-qubit CNOT depolarizing noise.
    """

    def __init__(self, p_1q: float = 0.0, p_cnot: float = 0.0):
        """
        Args:
            p_1q: Error probability for single-qubit gates.
            p_cnot: Error probability for two-qubit CNOT gates.
        """
        self.p_1q = float(p_1q)
        self.p_cnot = float(p_cnot)

        # Precompute 2-qubit non-identity Pauli matrices (15 matrices)
        self._2q_paulis = []
        labels = ["I", "X", "Y", "Z"]
        for c1, c2 in itertools.product(labels, labels):
            if c1 == "I" and c2 == "I":
                continue
            m = np.kron(PAULI_DICT[c1], PAULI_DICT[c2])
            self._2q_paulis.append(m)

        # 1-qubit non-identity Pauli matrices (3 matrices)
        self._1q_paulis = [X_GATE, Y_GATE, Z_GATE]

    def apply_1q_depolarizing(self, rho: np.ndarray, target: int, num_qubits: int) -> np.ndarray:
        """
        Applies single-qubit depolarizing noise on target qubit:
        E(rho) = (1 - p) rho + (p / 3) sum_{P in {X, Y, Z}} P_target rho P_target
        """
        if self.p_1q <= 0.0:
            return rho

        p = self.p_1q
        noisy_rho = (1.0 - p) * rho

        for p_mat in self._1q_paulis:
            # Embed p_mat into full n-qubit space
            ops = [I2] * num_qubits
            ops[target] = p_mat
            full_p = ops[0]
            for op in ops[1:]:
                full_p = np.kron(full_p, op)
            noisy_rho += (p / 3.0) * (full_p @ rho @ full_p.conj().T)

        return (noisy_rho + noisy_rho.conj().T) / 2.0

    def apply_cnot_depolarizing(
        self, rho: np.ndarray, control: int, target: int, num_qubits: int
    ) -> np.ndarray:
        """
        Applies two-qubit depolarizing noise on control and target qubits:
        E(rho) = (1 - p) rho + (p / 15) sum_{P in P_2 \ {II}} P_{c, t} rho P_{c, t}
        """
        if self.p_cnot <= 0.0:
            return rho

        p = self.p_cnot
        noisy_rho = (1.0 - p) * rho

        labels = ["I", "X", "Y", "Z"]
        for c1, c2 in itertools.product(labels, labels):
            if c1 == "I" and c2 == "I":
                continue

            # Embed 2-qubit Pauli into full n-qubit space
            ops = [I2] * num_qubits
            ops[control] = PAULI_DICT[c1]
            ops[target] = PAULI_DICT[c2]

            full_p = ops[0]
            for op in ops[1:]:
                full_p = np.kron(full_p, op)

            noisy_rho += (p / 15.0) * (full_p @ rho @ full_p.conj().T)

        # Enforce exact Hermiticity and unit trace
        noisy_rho = (noisy_rho + noisy_rho.conj().T) / 2.0
        trace = np.trace(noisy_rho)
        if abs(trace) > 1e-12:
            noisy_rho /= trace

        return noisy_rho
