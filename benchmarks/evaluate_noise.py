import os
import argparse
import json
import numpy as np
import torch

from envs.quantum_state_env import QuantumStateEnv
from envs.noise_models import NoiseModel
from models.dqn_agent import DQNAgent
from models.hybrid_q_agent import HybridDQNAgent
from utils.plotting import plot_noise_sweep


def evaluate_agent_on_noise_sweep(
    model_path: str,
    agent_type: str = "classical",
    num_qubits: int = 2,
    target_name: str = "bell_phi_plus",
    p_errors: list = None,
    num_eval_episodes: int = 15,
):
    if p_errors is None:
        p_errors = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]

    dummy_env = QuantumStateEnv(num_qubits=num_qubits, target_name=target_name)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n

    if agent_type == "hybrid":
        agent = HybridDQNAgent(state_dim=state_dim, action_dim=action_dim)
    else:
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    agent.load(model_path)

    fidelities = []
    purities = []
    cnot_counts = []
    histories = []

    for p in p_errors:
        eval_fids = []
        eval_purities = []
        eval_cnots = []
        best_history = []

        for _ in range(num_eval_episodes):
            noise_model = NoiseModel(p_cnot=p)
            # Channel capacity limit under 2-qubit depolarizing noise is ~ 1 - p
            target_fid = max(0.90, 1.0 - p - 0.01) if p > 0.0 else 0.99
            env = QuantumStateEnv(
                num_qubits=num_qubits,
                target_name=target_name,
                target_fidelity=target_fid,
                noise_model=noise_model,
                max_steps=12,
            )
            obs, info = env.reset()

            while True:
                action = agent.select_action(obs, evaluate=True)
                next_obs, reward, term, trunc, info = env.step(action)
                obs = next_obs
                if term or trunc:
                    break

            eval_fids.append(info["fidelity"])
            eval_purities.append(info["purity"])
            eval_cnots.append(info["cnot_count"])
            best_history = info["history"]

        fidelities.append(float(np.mean(eval_fids)))
        purities.append(float(np.mean(eval_purities)))
        cnot_counts.append(float(np.mean(eval_cnots)))
        histories.append(best_history)

    return {
        "p_errors": p_errors,
        "fidelities": fidelities,
        "purities": purities,
        "cnot_counts": cnot_counts,
        "histories": histories,
    }


def run_full_benchmark(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    p_errors = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]

    models_to_eval = [
        ("Classical Noiseless", "results/dqn_noiseless.pt", "classical"),
        ("Classical Noise-Aware", "results/dqn_noise_aware.pt", "classical"),
        ("Hybrid Q-Noiseless", "results/hybrid_noiseless.pt", "hybrid"),
        ("Hybrid Q-Noise-Aware", "results/hybrid_noise_aware.pt", "hybrid"),
    ]

    all_results = {}
    print("=" * 80)
    print("Evaluating Classical vs Hybrid Agents Across CNOT Noise Spectrum")
    print(f"Noise levels p_cnot: {p_errors}")
    print("=" * 80)

    for label, path, a_type in models_to_eval:
        if os.path.exists(path):
            print(f"Evaluating {label} ({path})...")
            res = evaluate_agent_on_noise_sweep(
                path, agent_type=a_type, num_qubits=2, target_name="bell_phi_plus", p_errors=p_errors
            )
            all_results[label] = res

    # Comparison Table
    print("\n### Complete Benchmark Table (Fidelity across CNOT error rates):")
    header = f"| {'p_cnot':^8} | " + " | ".join(f"{k:^20}" for k in all_results.keys()) + " |"
    sep = f"|{'-'*10}|" + "|".join(f"{'-'*22}" for _ in all_results.keys()) + "|"
    print(header)
    print(sep)

    for i, p in enumerate(p_errors):
        row_vals = [f"{p:^8.2f}"]
        for k in all_results.keys():
            fid = all_results[k]["fidelities"][i]
            row_vals.append(f"{fid:^20.4f}")
        print("| " + " | ".join(row_vals) + " |")

    # Save Results & Plots
    json_path = os.path.join(output_dir, "full_noise_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    plot_path = os.path.join(output_dir, "noise_robustness.png")
    plot_noise_sweep(all_results, save_path=plot_path)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    run_full_benchmark()
