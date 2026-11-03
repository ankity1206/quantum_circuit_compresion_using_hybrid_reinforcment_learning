import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from utils.quantum_metrics import (
    embed_1q_gate,
    cnot_matrix,
    rx,
    ry,
    rz,
    H_GATE,
    X_GATE,
    state_to_density_matrix,
    state_fidelity,
    purity,
    density_matrix_to_pauli_features,
    get_bell_state,
    get_ghz_state,
)
from envs.noise_models import NoiseModel


class QuantumStateEnv(gym.Env):
    """
    Gymnasium environment for quantum state synthesis and circuit optimization
    under open quantum system noise (CNOT depolarizing noise).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_qubits: int = 2,
        target_state: Optional[np.ndarray] = None,
        target_name: str = "bell_phi_plus",
        max_steps: int = 15,
        target_fidelity: float = 0.99,
        noise_model: Optional[NoiseModel] = None,
        gate_penalty: float = 0.02,
        cnot_penalty: float = 0.06,
        depth_discount: float = 0.95,
        bonus_reward: float = 10.0,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        self.max_steps = max_steps
        self.target_fidelity = target_fidelity
        self.gate_penalty = gate_penalty
        self.cnot_penalty = cnot_penalty
        self.depth_discount = depth_discount
        self.bonus_reward = bonus_reward

        # Noise model
        self.noise_model = noise_model if noise_model is not None else NoiseModel(p_cnot=0.0)

        # Set target state
        self.target_name = target_name
        if target_state is not None:
            self.target_psi = target_state
        else:
            if target_name == "bell_phi_plus":
                self.target_psi = get_bell_state("phi_plus")
            elif target_name == "ghz":
                self.target_psi = get_ghz_state(num_qubits)
            else:
                self.target_psi = get_bell_state("phi_plus")

        self.target_density = state_to_density_matrix(self.target_psi)

        # Build action dictionary
        self._build_action_space()

        # Observation space: Pauli expectation vector (4^n - 1) + normalized step count (1)
        self.obs_dim = (4**self.num_qubits - 1) + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # State tracking
        self.current_step = 0
        self.rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.prev_fidelity = 0.0
        self.history: List[str] = []
        self.cnot_count = 0

    def _build_action_space(self):
        """Constructs discrete actions and precomputes their unitaries."""
        self.actions_info = []

        # 1-qubit gates on each qubit
        single_gate_defs = [
            ("H", H_GATE),
            ("X", X_GATE),
            ("Rx_pi2", rx(np.pi / 2.0)),
            ("Ry_pi2", ry(np.pi / 2.0)),
            ("Rz_pi2", rz(np.pi / 2.0)),
            ("Rx_neg_pi2", rx(-np.pi / 2.0)),
            ("Ry_neg_pi2", ry(-np.pi / 2.0)),
        ]

        for q in range(self.num_qubits):
            for name, mat in single_gate_defs:
                unitary = embed_1q_gate(mat, q, self.num_qubits)
                self.actions_info.append({
                    "name": f"{name}(q{q})",
                    "type": "1q",
                    "target": q,
                    "unitary": unitary,
                })

        # 2-qubit CNOT gates between all directed pairs
        for c in range(self.num_qubits):
            for t in range(self.num_qubits):
                if c != t:
                    u_cnot = cnot_matrix(c, t, self.num_qubits)
                    self.actions_info.append({
                        "name": f"CNOT(c{c}->t{t})",
                        "type": "2q",
                        "control": c,
                        "target": t,
                        "unitary": u_cnot,
                    })

        # STOP action to terminate circuit
        self.stop_action_idx = len(self.actions_info)
        self.actions_info.append({"name": "STOP", "type": "stop"})

        self.action_space = spaces.Discrete(len(self.actions_info))

    def _get_obs(self) -> np.ndarray:
        """Computes Pauli expectation values + step fraction."""
        pauli_feats = density_matrix_to_pauli_features(
            self.rho, self.num_qubits, exclude_identity=True
        )
        step_frac = np.array([self.current_step / self.max_steps], dtype=np.float32)
        return np.concatenate([pauli_feats, step_frac]).astype(np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Initial state |0...0><0...0|
        init_psi = np.zeros(self.dim, dtype=np.complex128)
        init_psi[0] = 1.0
        self.rho = state_to_density_matrix(init_psi)

        self.current_step = 0
        self.cnot_count = 0
        self.history = []
        self.init_fidelity = state_fidelity(self.rho, self.target_psi)
        self.prev_fidelity = self.init_fidelity

        obs = self._get_obs()
        info = {
            "fidelity": self.prev_fidelity,
            "purity": purity(self.rho),
            "step": 0,
            "cnot_count": 0,
            "history": self.history,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        act_info = self.actions_info[action]
        act_type = act_info["type"]
        self.current_step += 1

        terminated = False
        truncated = False
        reward = 0.0

        if act_type == "stop":
            terminated = True
            self.history.append("STOP")
        else:
            prev_rho = self.rho.copy()
            u = act_info["unitary"]
            # Unitary evolution: rho' = U rho U†
            self.rho = u @ self.rho @ u.conj().T

            if act_type == "1q":
                self.history.append(act_info["name"])
                # Apply 1q depolarizing noise
                self.rho = self.noise_model.apply_1q_depolarizing(
                    self.rho, act_info["target"], self.num_qubits
                )
                # Base 1-qubit gate penalty
                reward -= self.gate_penalty
            elif act_type == "2q":
                self.cnot_count += 1
                self.history.append(act_info["name"])
                # Differentiated 2-qubit CNOT cost: gate_penalty + cnot_penalty
                reward -= (self.gate_penalty + self.cnot_penalty)
                # Apply 2q CNOT depolarizing noise
                self.rho = self.noise_model.apply_cnot_depolarizing(
                    self.rho, act_info["control"], act_info["target"], self.num_qubits
                )

        current_fidelity = state_fidelity(self.rho, self.target_psi)

        # Potential-based reward shaping: differential fidelity progress
        delta_fidelity = current_fidelity - self.prev_fidelity
        reward += delta_fidelity

        # Depth decay discount factor: gamma^depth
        depth_decay = self.depth_discount ** self.current_step

        # Fidelity gain above initial state baseline
        fid_gain = max(0.0, current_fidelity - self.init_fidelity)

        # Success condition
        if current_fidelity >= self.target_fidelity:
            terminated = True
            # Substantial bonus with depth discount gamma^depth to reward minimal gates
            reward += self.bonus_reward * (current_fidelity ** 2) * depth_decay
        elif terminated:  # Agent chose STOP before reaching threshold
            reward += self.bonus_reward * (fid_gain ** 2) * depth_decay

        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated:
                reward += self.bonus_reward * 0.5 * (fid_gain ** 2) * depth_decay

        self.prev_fidelity = current_fidelity
        obs = self._get_obs()

        info = {
            "fidelity": current_fidelity,
            "purity": purity(self.rho),
            "step": self.current_step,
            "cnot_count": self.cnot_count,
            "history": self.history,
            "success": bool(current_fidelity >= self.target_fidelity),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "ansi") -> str:
        s = f"Step {self.current_step}/{self.max_steps} | Fidelity: {self.prev_fidelity:.4f} | Purity: {purity(self.rho):.4f} | CNOTs: {self.cnot_count}\n"
        s += f"Gate History: {' -> '.join(self.history) if self.history else 'Initial'}"
        return s
