import os
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from envs.quantum_state_env import QuantumStateEnv
from envs.noise_models import NoiseModel
from models.dqn_agent import DQNAgent
from models.hybrid_q_agent import HybridDQNAgent
from utils.circuit_loader import (
    load_circuit_from_file_or_string,
    circuit_to_target_state,
    history_to_qiskit_circuit,
)
from train import train


def run_comparison(
    episodes: int = 250,
    p_cnot: float = 0.0,
    target_fidelity: float = 0.98,
    depth_discount: float = 0.95,
    save_dir: str = "results",
):
    os.makedirs(save_dir, exist_ok=True)
    circuit_name = "redundant_teleportation"
    qc = load_circuit_from_file_or_string(circuit_name)
    num_qubits = qc.num_qubits

    suffix = "with_discount" if depth_discount < 1.0 else "no_discount"
    print("=" * 70)
    print(f"BENCHMARK: STANDARD DQN vs. HYBRID DQN ({'WITH' if depth_discount < 1.0 else 'WITHOUT'} DEPTH DISCOUNT)")
    print(f"Target Circuit : {circuit_name} ({num_qubits} qubits)")
    print(f"Depth Discount : {depth_discount:.2f} ({'gamma^depth penalty active' if depth_discount < 1.0 else 'NO depth penalty'})")
    print(f"CNOT Noise     : p = {p_cnot:.4f}")
    print(f"Target Fidelity: {target_fidelity:.4f}")
    print("=" * 70)
    print("Input Target Circuit Diagram (Note the redundant CNOT pair on q1, q2):")
    print(qc.draw(output="text"))
    print("=" * 70)

    # 1. Train Standard (Classical) DQN
    model_name_c = f"dqn_{suffix}_teleport"
    print(f"\n>>> Phase 1: Training Standard Classical DQN ({model_name_c})...")
    agent_classic, metrics_classic = train(
        num_episodes=episodes,
        num_qubits=num_qubits,
        circuit_input=circuit_name,
        target_fidelity=target_fidelity,
        agent_type="classical",
        p_cnot=p_cnot,
        gate_penalty=0.01,
        cnot_penalty=0.04,
        depth_discount=depth_discount,
        max_steps=12,
        seed=42,
        save_dir=save_dir,
        model_name=model_name_c,
    )

    # 2. Train Hybrid DQN
    model_name_h = f"hybrid_{suffix}_teleport"
    print(f"\n>>> Phase 2: Training Hybrid Quantum DQN ({model_name_h})...")
    agent_hybrid, metrics_hybrid = train(
        num_episodes=episodes,
        num_qubits=num_qubits,
        circuit_input=circuit_name,
        target_fidelity=target_fidelity,
        agent_type="hybrid",
        p_cnot=p_cnot,
        gate_penalty=0.01,
        cnot_penalty=0.04,
        depth_discount=depth_discount,
        max_steps=12,
        seed=42,
        save_dir=save_dir,
        model_name=model_name_h,
    )

    # 3. Deterministic Greedy Evaluation
    print("\n" + "=" * 70)
    print("DETERMINISTIC EVALUATION COMPARISON")
    print("=" * 70)

    target_state = circuit_to_target_state(qc)
    noise_model = NoiseModel(p_cnot=p_cnot)

    def eval_agent(agent, label):
        env = QuantumStateEnv(
            num_qubits=num_qubits,
            target_state=target_state,
            target_fidelity=target_fidelity,
            noise_model=noise_model,
            max_steps=12,
        )
        obs, info = env.reset()
        while True:
            act = agent.select_action(obs, evaluate=True)
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        out_qc = history_to_qiskit_circuit(info["history"], num_qubits=num_qubits)
        return info, out_qc

    info_classic, qc_classic = eval_agent(agent_classic, "Standard Classical DQN")
    info_hybrid, qc_hybrid = eval_agent(agent_hybrid, "Hybrid Quantum DQN")

    # Display comparison table
    c_step = f"{info_classic['step']} gates"
    h_step = f"{info_hybrid['step']} gates"
    c_cnot = f"{info_classic['cnot_count']} CNOTs"
    h_cnot = f"{info_hybrid['cnot_count']} CNOTs"
    c_rem = "YES" if info_classic['cnot_count'] < 4 else "NO"
    h_rem = "YES" if info_hybrid['cnot_count'] < 4 else "NO"
    c_succ = "YES" if info_classic['success'] else "NO"
    h_succ = "YES" if info_hybrid['success'] else "NO"

    print(f"\n{'Metric':<25} | {'Input Circuit':<16} | {'Standard DQN':<16} | {'Hybrid DQN':<16}")
    print("-" * 80)
    print(f"{'Target Achieved?':<25} | {'N/A':<16} | {c_succ:<16} | {h_succ:<16}")
    print(f"{'Final Fidelity':<25} | {'1.0000':<16} | {info_classic['fidelity']:<16.4f} | {info_hybrid['fidelity']:<16.4f}")
    print(f"{'Total Gates':<25} | {'6 gates':<16} | {c_step:<16} | {h_step:<16}")
    print(f"{'Total CNOTs':<25} | {'4 CNOTs':<16} | {c_cnot:<16} | {h_cnot:<16}")
    print(f"{'Redundant CNOTs Removed?':<25} | {'NO (2 redundant)':<16} | {c_rem:<16} | {h_rem:<16}")
    print("=" * 80)

    print("\n--- Standard Classical DQN Synthesized Circuit ---")
    print(qc_classic.draw(output="text"))

    print("\n--- Hybrid Quantum DQN Synthesized Circuit ---")
    print(qc_hybrid.draw(output="text"))

    # Plot comparison curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Moving average
    def ma(arr, w=10):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    ep_arr = np.arange(1, len(metrics_classic["fidelity"]) + 1)
    sm_ep = ep_arr[9:] if len(ep_arr) >= 10 else ep_arr

    axes[0].plot(sm_ep, ma(metrics_classic["fidelity"]), label="Standard DQN", color="tab:blue", linewidth=2)
    axes[0].plot(sm_ep, ma(metrics_hybrid["fidelity"]), label="Hybrid DQN (PQC)", color="tab:purple", linewidth=2)
    axes[0].set_title("Target Fidelity vs Episode (No Depth Discount)", fontsize=12)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Fidelity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(sm_ep, ma(metrics_classic["cnot_count"]), label="Standard DQN", color="tab:blue", linewidth=2)
    axes[1].plot(sm_ep, ma(metrics_hybrid["cnot_count"]), label="Hybrid DQN (PQC)", color="tab:purple", linewidth=2)
    axes[1].set_title("CNOT Count vs Episode (Target: <= 2 CNOTs)", fontsize=12)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("CNOTs Used")
    axes[1].axhline(y=4, color="tab:red", linestyle="--", label="Input Bloated Circuit (4 CNOTs)")
    axes[1].axhline(y=2, color="tab:green", linestyle=":", label="Optimal Circuit (2 CNOTs)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"teleportation_comparison_{suffix}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nComparison plot saved to: {plot_path}")

    # Save summary report
    summary_report = {
        "circuit": circuit_name,
        "input_cnot_count": 4,
        "input_total_gates": 6,
        "depth_discount": depth_discount,
        "standard_dqn": {
            "fidelity": info_classic["fidelity"],
            "total_gates": info_classic["step"],
            "cnot_count": info_classic["cnot_count"],
            "history": info_classic["history"],
        },
        "hybrid_dqn": {
            "fidelity": info_hybrid["fidelity"],
            "total_gates": info_hybrid["step"],
            "cnot_count": info_hybrid["cnot_count"],
            "history": info_hybrid["history"],
        },
    }

    report_json_path = os.path.join(save_dir, f"teleportation_comparison_{suffix}_results.json")
    with open(report_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--fidelity", type=float, default=0.98)
    parser.add_argument("--depth_discount", type=float, default=0.95, help="Depth discount factor (e.g. 0.95)")
    args = parser.parse_args()

    run_comparison(
        episodes=args.episodes,
        p_cnot=args.noise,
        target_fidelity=args.fidelity,
        depth_discount=args.depth_discount,
    )
