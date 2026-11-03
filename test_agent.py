import argparse
import os
from typing import Optional
import numpy as np
import torch

from envs.quantum_state_env import QuantumStateEnv
from envs.noise_models import NoiseModel
from models.dqn_agent import DQNAgent
from models.hybrid_q_agent import HybridDQNAgent
from utils.circuit_loader import (
    load_circuit_from_file_or_string,
    circuit_to_target_state,
    history_to_qiskit_circuit,
)


def detect_model_type(model_path: str) -> str:
    """Auto-detects whether the checkpoint is classical or hybrid."""
    checkpoint = torch.load(model_path, map_location="cpu")
    policy_dict = checkpoint.get("policy_net", checkpoint)
    if any("q_layer" in k or "pre_net" in k for k in policy_dict.keys()):
        return "hybrid"
    return "classical"


def test_model(
    model_path: str,
    agent_type: str = "auto",
    circuit_input: Optional[str] = None,
    p_cnot: float = 0.0,
    target_name: str = "bell_phi_plus",
    target_fidelity: float = None,
    num_qubits: int = 2,
    max_steps: int = 10,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    target_state = None
    qc = None
    if circuit_input is not None:
        qc = load_circuit_from_file_or_string(circuit_input)
        num_qubits = qc.num_qubits
        target_state = circuit_to_target_state(qc)
        target_name = f"CustomCircuit({circuit_input})"

    # Auto-detect agent type if set to "auto"
    detected_type = detect_model_type(model_path)
    if agent_type != "auto" and agent_type != detected_type:
        print(f"[NOTE] Overriding user requested type '{agent_type}' -> '{detected_type}' (detected from checkpoint weights)")
    agent_type = detected_type

    # Adjust target fidelity threshold to physical channel limit if noise is present
    if target_fidelity is None:
        target_fidelity = 0.95 if p_cnot > 0.0 else 0.99

    # Build environment
    noise_model = NoiseModel(p_cnot=p_cnot)
    env = QuantumStateEnv(
        num_qubits=num_qubits,
        target_state=target_state,
        target_name=target_name,
        target_fidelity=target_fidelity,
        noise_model=noise_model,
        max_steps=max_steps,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load agent
    if agent_type == "hybrid":
        agent = HybridDQNAgent(state_dim=state_dim, action_dim=action_dim)
    else:
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    agent.load(model_path)
    print("=" * 65)
    print(f"Loaded Model : {model_path} ({agent_type.upper()})")
    print(f"Target State : {target_name} ({num_qubits} qubits)")
    print(f"CNOT Noise   : p = {p_cnot:.4f}")
    if qc is not None:
        print("-" * 65)
        print("Input Target Circuit Diagram:")
        print(qc.draw(output="text"))
    print("=" * 65)

    # Run greedy rollout
    obs, info = env.reset()
    step_num = 0
    print(f"Step {step_num:2d} | Initial State | Fidelity: {info['fidelity']:.4f} | Purity: {info['purity']:.4f}")

    while True:
        # evaluate=True uses strictly greedy policy (no random epsilon exploration)
        action = agent.select_action(obs, evaluate=True)
        act_info = env.actions_info[action]

        next_obs, reward, term, trunc, info = env.step(action)
        step_num += 1

        print(
            f"Step {step_num:2d} | Action: {act_info['name']:<18} | "
            f"Fidelity: {info['fidelity']:.4f} | Purity: {info['purity']:.4f} | Rew: {reward:+.3f}"
        )

        obs = next_obs
        if term or trunc:
            break

    print("=" * 65)
    print("Execution Summary:")
    print(f"  Target Achieved?    : {'YES (Success!)' if info['success'] else 'NO'}")
    print(f"  Final Fidelity      : {info['fidelity']:.4f}")
    print(f"  Final Purity        : {info['purity']:.4f}")
    print(f"  Total Gates Applied : {info['step']}")
    print(f"  Total CNOTs Used    : {info['cnot_count']}")
    print(f"  Gate Sequence       : {' -> '.join(info['history'])}")
    print("-" * 65)

    # Render Qiskit Circuit Diagram for the output sequence
    out_qc = history_to_qiskit_circuit(info["history"], num_qubits=num_qubits)
    print("Synthesized Output Circuit Diagram:")
    print(out_qc.draw(output="text"))
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained quantum RL agent.")
    parser.add_argument(
        "--circuit",
        type=str,
        default=None,
        help="Custom input circuit (.qasm file path or template 'bell', 'redundant_bell', etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/hybrid_noise_aware.pt",
        help="Path to .pt model weights",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "classical", "hybrid"],
        help="Agent architecture ('auto', 'classical', or 'hybrid')",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="CNOT depolarizing error rate p (e.g. 0.0 or 0.04)",
    )
    parser.add_argument(
        "--target_fidelity",
        "--threshold",
        type=float,
        default=None,
        help="Custom target fidelity success threshold (defaults to physical capacity 1 - p)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="bell_phi_plus",
        help="Target quantum state",
    )
    args = parser.parse_args()

    test_model(
        model_path=args.model,
        agent_type=args.type,
        circuit_input=args.circuit,
        p_cnot=args.noise,
        target_name=args.target,
        target_fidelity=args.target_fidelity,
    )
