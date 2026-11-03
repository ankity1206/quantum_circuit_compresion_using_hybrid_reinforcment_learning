import itertools
import numpy as np
import scipy.linalg

# Standard 1-qubit Pauli matrices and basic gates
I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
X_GATE = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y_GATE = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z_GATE = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H_GATE = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
S_GATE = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
T_GATE = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128)

PAULI_DICT = {"I": I2, "X": X_GATE, "Y": Y_GATE, "Z": Z_GATE}


def rx(theta: float) -> np.ndarray:
    """Single-qubit Rx rotation unitary."""
    half = theta / 2.0
    return np.cos(half) * I2 - 1.0j * np.sin(half) * X_GATE


def ry(theta: float) -> np.ndarray:
    """Single-qubit Ry rotation unitary."""
    half = theta / 2.0
    return np.cos(half) * I2 - 1.0j * np.sin(half) * Y_GATE


def rz(theta: float) -> np.ndarray:
    """Single-qubit Rz rotation unitary."""
    half = theta / 2.0
    return np.cos(half) * I2 - 1.0j * np.sin(half) * Z_GATE


def cnot_matrix(control: int, target: int, num_qubits: int) -> np.ndarray:
    """
    Constructs an n-qubit unitary matrix for CNOT with given control and target.
    Qubit index 0 is the most significant qubit in tensor product convention (or standard convention).
    """
    dim = 2**num_qubits
    u = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        # Check if control bit is set (bit index from left: num_qubits - 1 - control)
        is_control_set = (i >> (num_qubits - 1 - control)) & 1
        if is_control_set:
            # Flip target bit
            j = i ^ (1 << (num_qubits - 1 - target))
        else:
            j = i
        u[j, i] = 1.0
    return u


def embed_1q_gate(gate: np.ndarray, target: int, num_qubits: int) -> np.ndarray:
    """
    Embeds a 1-qubit gate into an n-qubit Hilbert space via Kronecker products.
    """
    ops = [I2] * num_qubits
    ops[target] = gate
    full_op = ops[0]
    for op in ops[1:]:
        full_op = np.kron(full_op, op)
    return full_op


def state_to_density_matrix(psi: np.ndarray) -> np.ndarray:
    """Converts a pure statevector |psi> to density matrix |psi><psi|."""
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1, 1)
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm
    return psi @ psi.conj().T


def state_fidelity(rho: np.ndarray, target_state: np.ndarray) -> float:
    """
    Computes fidelity between a mixed state rho and a target state.
    If target_state is 1D (pure state |psi>), uses F = <psi|rho|psi>.
    If target_state is 2D (mixed state sigma), uses Uhlmann fidelity:
      (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    """
    if target_state.ndim == 1 or (target_state.ndim == 2 and target_state.shape[1] == 1):
        psi = target_state.reshape(-1, 1)
        fid = np.real((psi.conj().T @ rho @ psi)[0, 0])
        return float(np.clip(fid, 0.0, 1.0))

    # General Uhlmann fidelity for two density matrices
    sqrt_rho = scipy.linalg.sqrtm(rho)
    temp = sqrt_rho @ target_state @ sqrt_rho
    sqrt_temp = scipy.linalg.sqrtm(temp)
    fid = float(np.real(np.trace(sqrt_temp)) ** 2)
    return float(np.clip(fid, 0.0, 1.0))


def purity(rho: np.ndarray) -> float:
    """Computes purity Tr(rho^2) of density matrix."""
    return float(np.real(np.trace(rho @ rho)))


def pauli_basis_strings(num_qubits: int):
    """Generates all 4^num_qubits Pauli strings and their matrices."""
    labels = ["I", "X", "Y", "Z"]
    for combo in itertools.product(labels, repeat=num_qubits):
        name = "".join(combo)
        matrix = PAULI_DICT[combo[0]]
        for c in combo[1:]:
            matrix = np.kron(matrix, PAULI_DICT[c])
        yield name, matrix


def density_matrix_to_pauli_features(rho: np.ndarray, num_qubits: int, exclude_identity: bool = True) -> np.ndarray:
    """
    Maps an n-qubit density matrix rho to its Pauli expectation vector <P_i> = Tr(rho P_i).
    These expectation values are strictly real and bounded in [-1, 1].
    Length: 4^n - 1 (if excluding identity Tr(rho)=1) or 4^n.
    For n=2: 15 non-trivial real features.
    """
    features = []
    labels = ["I", "X", "Y", "Z"]
    for combo in itertools.product(labels, repeat=num_qubits):
        if exclude_identity and all(c == "I" for c in combo):
            continue
        matrix = PAULI_DICT[combo[0]]
        for c in combo[1:]:
            matrix = np.kron(matrix, PAULI_DICT[c])
        exp_val = np.real(np.trace(rho @ matrix))
        features.append(exp_val)
    return np.array(features, dtype=np.float32)


# Target State Generators
def get_bell_state(bell_type: str = "phi_plus") -> np.ndarray:
    """Returns pure 2-qubit Bell statevector."""
    if bell_type == "phi_plus":
        # (|00> + |11>) / sqrt(2)
        state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    elif bell_type == "phi_minus":
        # (|00> - |11>) / sqrt(2)
        state = np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2)
    elif bell_type == "psi_plus":
        # (|01> + |10>) / sqrt(2)
        state = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2)
    elif bell_type == "psi_minus":
        # (|01> - |10>) / sqrt(2)
        state = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown Bell state type: {bell_type}")
    return state


def get_ghz_state(num_qubits: int = 3) -> np.ndarray:
    """Returns pure n-qubit GHZ statevector (|00...0> + |11...1>) / sqrt(2)."""
    dim = 2**num_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)
    state[-1] = 1.0 / np.sqrt(2)
    return state


def get_w_state(num_qubits: int = 3) -> np.ndarray:
    """Returns pure n-qubit W statevector (|001> + |010> + |100>) / sqrt(3)."""
    dim = 2**num_qubits
    state = np.zeros(dim, dtype=np.complex128)
    norm = 1.0 / np.sqrt(num_qubits)
    for q in range(num_qubits):
        idx = 1 << (num_qubits - 1 - q)
        state[idx] = norm
    return state
