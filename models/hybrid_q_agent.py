import random
from collections import deque
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml


class HybridQuantumDuelingQNetwork(nn.Module):
    """
    Hybrid Quantum-Classical Dueling Q-Network.
    Architecture:
      Classical Pre-Net (Linear + Tanh * pi)
        -> PennyLane Parameterized Quantum Circuit (PQC with StronglyEntanglingLayers)
        -> Pauli-Z Expectation values & Correlators
        -> Dueling Value & Advantage heads
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        agent_qubits: int = 4,
        ansatz_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_qubits = agent_qubits
        self.ansatz_layers = ansatz_layers

        # 1. Classical state encoding to agent qubits
        self.pre_net = nn.Sequential(
            nn.Linear(state_dim, agent_qubits),
            nn.Tanh(),
        )

        # 2. Parameterized Quantum Circuit
        dev = qml.device("lightning.qubit", wires=agent_qubits)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def pqc_circuit(inputs, weights):
            # Scale input angles to [-pi, pi]
            scaled_inputs = inputs * np.pi
            qml.AngleEmbedding(scaled_inputs, wires=range(agent_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(agent_qubits))

            # Observable readout: 1-qubit Z expectations + 2-qubit correlators
            obs = [qml.expval(qml.PauliZ(i)) for i in range(agent_qubits)]
            for i in range(agent_qubits):
                j = (i + 1) % agent_qubits
                obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
            return obs

        # Weight shape for StronglyEntanglingLayers: (ansatz_layers, agent_qubits, 3)
        weight_shapes = {"weights": (ansatz_layers, agent_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(pqc_circuit, weight_shapes)

        # Quantum feature dimension: m single-qubit Z + m correlators
        q_feat_dim = agent_qubits * 2

        # 3. Dueling Heads
        # Value stream: V(s)
        self.value_head = nn.Sequential(
            nn.Linear(q_feat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_head = nn.Sequential(
            nn.Linear(q_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Pre-process state features
        encoded_angles = self.pre_net(state)

        # Quantum layer forward pass
        q_features = self.q_layer(encoded_angles)

        # Dueling aggregation
        values = self.value_head(q_features)
        advantages = self.advantage_head(q_features)

        q_vals = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_vals


class HybridDQNAgent:
    """
    Hybrid Quantum Deep Q-Learning Agent using PennyLane PQC.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        agent_qubits: int = 4,
        ansatz_layers: int = 2,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 30000,
        batch_size: int = 32,
        target_update_freq: int = 30,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.03,
        epsilon_decay_steps: int = 800,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Policy and Target Hybrid Quantum Networks
        self.policy_net = HybridQuantumDuelingQNetwork(
            state_dim, action_dim, agent_qubits, ansatz_layers
        ).to(self.device)

        self.target_net = HybridQuantumDuelingQNetwork(
            state_dim, action_dim, agent_qubits, ansatz_layers
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        from models.dqn_agent import ReplayBuffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / float(epsilon_decay_steps)
        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=-1).item())

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q estimates
        curr_q = self.policy_net(states).gather(1, actions)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=-1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_steps += 1

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        return float(loss.item())

    def save(self, filepath: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.total_steps = checkpoint.get("total_steps", 0)
