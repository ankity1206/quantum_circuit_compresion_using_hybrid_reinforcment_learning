#!/usr/bin/env python3
"""
OpenQASM Export & Verification Tool for Quantum Reinforcement Learning.

Serializes RL-synthesized quantum circuits into standard OpenQASM 2.0 or 3.0 files (.qasm),
and provides automated closed-loop verification against target states under open-system noise.
"""

import os
import sys
import argparse
import re
from typing import Optional, List, Tuple
import numpy as np
import torch
from qiskit import QuantumCircuit, qasm2, qasm3

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.quantum_state_env import QuantumStateEnv
from envs.noise_models import NoiseModel
from models.dqn_agent import DQNAgent
from models.hybrid_q_agent import HybridDQNAgent
from models.scaled_hybrid_q_agent import ScaledHybridDQNAgent
from models.adaptive_hybrid_q_agent import AdaptiveHybridDQNAgent
from utils.circuit_loader import (
    load_circuit_from_file_or_string,
    circuit_to_target_state,
    history_to_qiskit_circuit,
)


def load_universal_agent(model_path: str, state_dim: int, action_dim: int, num_target_qubits: int):
    """
    Auto-detects and loads any trained checkpoint:
      - Classical DQN (MLP)
      - Baseline 4Q Hybrid DQN
      - Scaled 6Q/8Q Hybrid DQN
      - Adaptive Hybrid DQN
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    policy_dict = checkpoint.get("policy_net", checkpoint)

    if "num_target_qubits" in checkpoint:
        agent = AdaptiveHybridDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_target_qubits=num_target_qubits,
            agent_qubits=checkpoint.get("agent_qubits", 4),
            ansatz_layers=checkpoint.get("ansatz_layers", 2),
            auto_size=False,
        )
        model_type = f"Adaptive Hybrid ({checkpoint.get('agent_qubits')}Q, {checkpoint.get('ansatz_layers')}L)"
    elif "agent_qubits" in checkpoint:
        agent = ScaledHybridDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_qubits=checkpoint["agent_qubits"],
            ansatz_layers=checkpoint["ansatz_layers"],
        )
        model_type = f"Scaled Hybrid ({checkpoint['agent_qubits']}Q, {checkpoint['ansatz_layers']}L)"
    elif any("q_layer" in k or "pre_net" in k for k in policy_dict.keys()):
        agent = HybridDQNAgent(state_dim=state_dim, action_dim=action_dim)
        model_type = "Baseline Hybrid (4Q, 2L)"
    else:
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        model_type = "Classical DQN (MLP)"

    agent.load(model_path)
    return agent, model_type


def synthesize_circuit_from_model(
    model_path: str,
    circuit_input: str,
    max_steps: int = 15,
) -> Tuple[QuantumCircuit, List[str], dict, str]:
    """
    Runs a greedy deterministic rollout on the target environment using the trained model,
    returning the synthesized Qiskit circuit and rollout metrics.
    """
    qc_target = load_circuit_from_file_or_string(circuit_input)
    num_qubits = qc_target.num_qubits
    target_state = circuit_to_target_state(qc_target)

    # Clean noiseless environment for exact synthesis trace
    env = QuantumStateEnv(
        num_qubits=num_qubits,
        target_state=target_state,
        target_name=circuit_input,
        target_fidelity=0.99,
        noise_model=NoiseModel(p_cnot=0.0),
        max_steps=max_steps,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent, model_type = load_universal_agent(model_path, state_dim, action_dim, num_qubits)

    obs, info = env.reset()
    while True:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break

    out_qc = history_to_qiskit_circuit(info["history"], num_qubits=num_qubits)
    return out_qc, info["history"], info, model_type


def export_to_qasm_string(qc: QuantumCircuit, qasm_version: str = "qasm2") -> str:
    """Serializes a Qiskit QuantumCircuit into OpenQASM 2.0 or 3.0 string."""
    if qasm_version.lower() in ["qasm2", "2", "2.0"]:
        return qasm2.dumps(qc)
    elif qasm_version.lower() in ["qasm3", "3", "3.0"]:
        return qasm3.dumps(qc)
    else:
        raise ValueError(f"Unsupported OpenQASM format '{qasm_version}'. Choose 'qasm2' or 'qasm3'.")


def verify_exported_qasm(
    qasm_path: str,
    target_state: np.ndarray,
    p_cnot: float = 0.0,
) -> dict:
    """
    Reloads the exported QASM file and verifies its physical fidelity and gate count
    in the open-system density matrix environment.
    """
    qc_reloaded = load_circuit_from_file_or_string(qasm_path)
    num_qubits = qc_reloaded.num_qubits
    env = QuantumStateEnv(
        num_qubits=num_qubits,
        target_state=target_state,
        target_name="QASM_Verification",
        noise_model=NoiseModel(p_cnot=p_cnot),
    )

    # Execute gates from reloaded circuit
    action_map = {act["name"]: idx for idx, act in enumerate(env.actions_info)}
    obs, info = env.reset()
    cnot_count = 0
    gate_count = 0

    for instruction in qc_reloaded.data:
        op = instruction.operation
        qargs = instruction.qubits
        gate_name = op.name.lower()
        gate_count += 1

        if gate_name in ["cx", "cnot"]:
            cnot_count += 1
            c = qc_reloaded.find_bit(qargs[0]).index
            t = qc_reloaded.find_bit(qargs[1]).index
            act_name = f"CNOT(c{c}->t{t})"
        elif gate_name == "h":
            q = qc_reloaded.find_bit(qargs[0]).index
            act_name = f"H(q{q})"
        elif gate_name == "x":
            q = qc_reloaded.find_bit(qargs[0]).index
            act_name = f"X(q{q})"
        elif gate_name == "rx":
            q = qc_reloaded.find_bit(qargs[0]).index
            theta = float(op.params[0])
            act_name = f"Rx_pi2(q{q})" if theta > 0 else f"Rx_neg_pi2(q{q})"
        elif gate_name == "ry":
            q = qc_reloaded.find_bit(qargs[0]).index
            theta = float(op.params[0])
            act_name = f"Ry_pi2(q{q})" if theta > 0 else f"Ry_neg_pi2(q{q})"
        elif gate_name == "rz":
            q = qc_reloaded.find_bit(qargs[0]).index
            theta = float(op.params[0])
            act_name = f"Rz_pi2(q{q})" if theta > 0 else f"Rz_neg_pi2(q{q})"
        else:
            act_name = None

        if act_name in action_map:
            act_idx = action_map[act_name]
            obs, reward, term, trunc, info = env.step(act_idx)

    return {
        "verified_fidelity": float(info["fidelity"]),
        "verified_purity": float(info["purity"]),
        "total_gates": gate_count,
        "cnot_count": cnot_count,
        "target_reached": bool(info["success"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export RL-synthesized quantum circuits to OpenQASM (.qasm) format with automated testing."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model weights (.pt). If omitted, exports target circuit directly.",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="redundant_teleportation",
        help="Target circuit preset ('bell', 'redundant_bell', 'ghz3', 'w3', 'redundant_teleportation') or .qasm file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Destination path for .qasm file. If omitted, prints to standard output or saves to results/.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="qasm2",
        choices=["qasm2", "qasm3"],
        help="OpenQASM specification format (default: 'qasm2').",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Automatically reload and verify the exported .qasm file in the density matrix environment.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="CNOT depolarizing error rate for verification (default: 0.0).",
    )
    args = parser.parse_args()

    # Load target circuit
    qc_target = load_circuit_from_file_or_string(args.circuit)
    target_state = circuit_to_target_state(qc_target)
    num_qubits = qc_target.num_qubits

    print("=" * 70)
    print("OPENQASM EXPORT & TESTING TOOL")
    print(f"Target Plant : {args.circuit} ({num_qubits} qubits)")
    print(f"QASM Format  : {args.format.upper()}")
    print("=" * 70)

    # 1. Synthesize or Extract Circuit
    if args.model:
        print(f">>> Synthesizing circuit using policy checkpoint: {args.model}")
        qc_out, history, info, model_type = synthesize_circuit_from_model(args.model, args.circuit)
        print(f"Detected Model Type : {model_type}")
        print(f"Synthesized Sequence: {' -> '.join(history)}")
        print(f"Fidelity on Plant   : {info['fidelity']:.4f}")
        print(f"Total Gates         : {info['step']} (CNOTs: {info['cnot_count']})")
    else:
        print(f">>> Direct export of target circuit: {args.circuit}")
        qc_out = qc_target

    print("-" * 70)
    print("Circuit Diagram:")
    print(qc_out.draw(output="text"))
    print("-" * 70)

    # 2. Convert to QASM string
    qasm_str = export_to_qasm_string(qc_out, qasm_version=args.format)

    # 3. Save to file
    out_path = args.out
    if out_path is None:
        clean_name = os.path.splitext(os.path.basename(args.circuit))[0]
        prefix = "synthesized" if args.model else "target"
        out_path = os.path.join("results", f"{prefix}_{clean_name}_{args.format}.qasm")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(qasm_str)

    print(f"Successfully exported OpenQASM circuit to: {out_path}")
    print("OpenQASM Preview:")
    for line in qasm_str.strip().split("\n")[:10]:
        print(f"  {line}")
    if len(qasm_str.strip().split("\n")) > 10:
        print("  ...")
    print("=" * 70)

    # 4. Automated Closed-Loop Verification
    if args.verify:
        print(">>> Performing Closed-Loop Verification on Exported QASM File...")
        verif = verify_exported_qasm(out_path, target_state, p_cnot=args.noise)
        print(f"  Reloaded QASM File : {out_path}")
        print(f"  CNOT Noise Level   : p = {args.noise:.4f}")
        print(f"  Verified Fidelity  : {verif['verified_fidelity']:.4f}")
        print(f"  Verified Purity    : {verif['verified_purity']:.4f}")
        print(f"  Verified Gates     : {verif['total_gates']} (CNOTs: {verif['cnot_count']})")
        print(f"  Target Achieved?   : {'YES (PASS)' if verif['verified_fidelity'] >= 0.95 else 'NO (FAIL)'}")
        print("=" * 70)


if __name__ == "__main__":
    main()
