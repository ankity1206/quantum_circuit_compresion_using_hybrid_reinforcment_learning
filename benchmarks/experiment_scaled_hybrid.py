import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from envs.quantum_state_env import QuantumStateEnv
from envs.noise_models import NoiseModel
from models.dqn_agent import DQNAgent
from models.hybrid_q_agent import HybridDQNAgent
from models.scaled_hybrid_q_agent import ScaledHybridDQNAgent
from utils.circuit_loader import (
    load_circuit_from_file_or_string,
    circuit_to_target_state,
    history_to_qiskit_circuit,
)


def train_agent_generic(
    agent,
    env,
    num_episodes: int,
    model_name: str,
    save_dir: str = "results",
):
    metrics = {
        "episode": [],
        "reward": [],
        "fidelity": [],
        "purity": [],
        "cnot_count": [],
        "steps": [],
        "success": [],
        "history": [],
    }

    start_time = time.time()
    agent_step = 0

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0

        while True:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            agent_step += 1

            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            if agent_step % 2 == 0:
                agent.update()

            ep_reward += reward
            obs = next_obs

            if done:
                break

        metrics["episode"].append(ep)
        metrics["reward"].append(float(ep_reward))
        metrics["fidelity"].append(float(info["fidelity"]))
        metrics["purity"].append(float(info["purity"]))
        metrics["cnot_count"].append(int(info["cnot_count"]))
        metrics["steps"].append(int(info["step"]))
        metrics["success"].append(bool(info["success"]))
        metrics["history"].append(info["history"])

        if ep % 25 == 0 or ep == 1:
            avg_fid = np.mean(metrics["fidelity"][-25:])
            avg_cnots = np.mean(metrics["cnot_count"][-25:])
            success_rate = np.mean(metrics["success"][-25:]) * 100
            print(
                f"[{model_name}] Ep {ep:4d}/{num_episodes} | Rew: {ep_reward:6.2f} | "
                f"Fid (last 25): {avg_fid:.4f} | CNOTs: {avg_cnots:.1f} | "
                f"Success: {success_rate:5.1f}% | Eps: {agent.epsilon:.3f}"
            )

    elapsed = time.time() - start_time
    print(f"[{model_name}] Completed in {elapsed:.1f}s.")

    # Save model and metrics
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    agent.save(model_path)

    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def eval_agent(agent, env):
    obs, info = env.reset()
    while True:
        act = agent.select_action(obs, evaluate=True)
        obs, r, term, trunc, info = env.step(act)
        if term or trunc:
            break
    return info


def run_experiment(
    episodes: int = 200,
    circuit_name: str = "redundant_teleportation",
    depth_discount: float = 0.95,
    p_cnot: float = 0.0,
    target_fidelity: float = 0.98,
    agent_qubits: int = 6,
    ansatz_layers: int = 3,
    use_data_reuploading: bool = True,
    only_scaled: bool = False,
    save_dir: str = "results",
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    qc = load_circuit_from_file_or_string(circuit_name)
    num_qubits = qc.num_qubits
    target_state = circuit_to_target_state(qc)
    noise_model = NoiseModel(p_cnot=p_cnot)

    print("=" * 70)
    print("SCALED HYBRID Q-RL EXPERIMENT")
    print(f"Target Circuit      : {circuit_name} ({num_qubits} qubits)")
    print(f"Episodes            : {episodes}")
    print(f"Depth Discount      : {depth_discount}")
    print(f"Controller Qubits   : {agent_qubits}")
    print(f"Ansatz Layers       : {ansatz_layers}")
    print(f"Data Re-uploading   : {use_data_reuploading}")
    print("=" * 70)
    print("Input Target Circuit Diagram:")
    print(qc.draw(output="text"))
    print("=" * 70)

    # Helper to instantiate env
    def make_env():
        return QuantumStateEnv(
            num_qubits=num_qubits,
            target_state=target_state,
            target_name=circuit_name,
            target_fidelity=target_fidelity,
            noise_model=noise_model,
            gate_penalty=0.01,
            cnot_penalty=0.04,
            depth_discount=depth_discount,
            max_steps=12,
        )

    env_sample = make_env()
    state_dim = env_sample.observation_space.shape[0]
    action_dim = env_sample.action_space.n

    # Scaled Hybrid DQN Agent
    print(f"\n>>> Setting up Scaled {agent_qubits}-Qubit Hybrid DQN Agent...")
    scaled_hybrid_agent = ScaledHybridDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        agent_qubits=agent_qubits,
        ansatz_layers=ansatz_layers,
        use_data_reuploading=use_data_reuploading,
        correlator_type="all_pairs",
        hidden_dim=32,
        learning_rate=2e-3,
        gamma=0.99,
        buffer_capacity=30000,
        batch_size=32,
        target_update_freq=25,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay_steps=max(600, int(episodes * 3.5)),
    )
    breakdown_scaled = scaled_hybrid_agent.policy_net.get_parameter_breakdown()
    print("Scaled Hybrid Network Architecture:")
    for k, v in breakdown_scaled.items():
        print(f"  {k}: {v}")

    metrics_classic = None
    metrics_baseline = None

    if not only_scaled:
        # Baseline 4-Qubit Hybrid DQN Agent
        print("\n>>> Setting up Baseline 4-Qubit Hybrid DQN Agent...")
        baseline_hybrid_agent = HybridDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_qubits=4,
            ansatz_layers=2,
            learning_rate=2e-3,
            gamma=0.99,
            buffer_capacity=30000,
            batch_size=32,
            target_update_freq=25,
            epsilon_start=1.0,
            epsilon_end=0.03,
            epsilon_decay_steps=max(600, int(episodes * 3.5)),
        )

        # Standard Classical DQN Agent
        print("\n>>> Setting up Standard Classical DQN Agent...")
        classical_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=5e-4,
            gamma=0.99,
            buffer_capacity=30000,
            batch_size=64,
            target_update_freq=20,
            epsilon_start=1.0,
            epsilon_end=0.03,
            epsilon_decay_steps=max(500, int(episodes * 4)),
        )

        # Train Classical
        print("\n" + "=" * 70)
        print("PHASE 1: Training Standard Classical DQN...")
        print("=" * 70)
        metrics_classic = train_agent_generic(
            classical_agent, make_env(), episodes, "dqn_teleport_ref", save_dir
        )

        # Train Baseline Hybrid
        print("\n" + "=" * 70)
        print("PHASE 2: Training Baseline 4-Qubit Hybrid DQN...")
        print("=" * 70)
        metrics_baseline = train_agent_generic(
            baseline_hybrid_agent, make_env(), episodes, "hybrid_4q_baseline", save_dir
        )

    # Train Scaled Hybrid
    print("\n" + "=" * 70)
    print(f"PHASE: Training Scaled {agent_qubits}-Qubit Hybrid DQN...")
    print("=" * 70)
    scaled_model_name = f"hybrid_{agent_qubits}q_scaled"
    metrics_scaled = train_agent_generic(
        scaled_hybrid_agent, make_env(), episodes, scaled_model_name, save_dir
    )

    # Deterministic Evaluation
    print("\n" + "=" * 70)
    print("DETERMINISTIC GREEDY EVALUATION COMPARISON")
    print("=" * 70)

    eval_env = make_env()
    info_scaled = eval_agent(scaled_hybrid_agent, eval_env)
    print(f"Scaled Hybrid ({agent_qubits}Q) -> Fidelity: {info_scaled['fidelity']:.4f} | Gates: {info_scaled['step']} | CNOTs: {info_scaled['cnot_count']} | History: {info_scaled['history']}")

    summary = {
        "circuit": circuit_name,
        "episodes": episodes,
        "scaled_hybrid": {
            "agent_qubits": agent_qubits,
            "ansatz_layers": ansatz_layers,
            "use_data_reuploading": use_data_reuploading,
            "fidelity": float(info_scaled["fidelity"]),
            "gates": int(info_scaled["step"]),
            "cnots": int(info_scaled["cnot_count"]),
            "history": info_scaled["history"],
            "parameter_breakdown": breakdown_scaled,
        },
    }

    if not only_scaled:
        info_classic = eval_agent(classical_agent, eval_env)
        info_baseline = eval_agent(baseline_hybrid_agent, eval_env)
        print(f"Classical DQN    -> Fidelity: {info_classic['fidelity']:.4f} | Gates: {info_classic['step']} | CNOTs: {info_classic['cnot_count']} | History: {info_classic['history']}")
        print(f"4Q Hybrid DQN    -> Fidelity: {info_baseline['fidelity']:.4f} | Gates: {info_baseline['step']} | CNOTs: {info_baseline['cnot_count']} | History: {info_baseline['history']}")
        summary["classical_dqn"] = {
            "fidelity": float(info_classic["fidelity"]),
            "gates": int(info_classic["step"]),
            "cnots": int(info_classic["cnot_count"]),
            "history": info_classic["history"],
        }
        summary["hybrid_4q_baseline"] = {
            "fidelity": float(info_baseline["fidelity"]),
            "gates": int(info_baseline["step"]),
            "cnots": int(info_baseline["cnot_count"]),
            "history": info_baseline["history"],
        }

    summary_path = os.path.join(save_dir, "scaled_hybrid_teleport_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plot comparison
    def smooth(arr, w=15):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Fidelity
    if metrics_classic:
        axes[0].plot(smooth(metrics_classic["fidelity"]), label="Classical DQN", color="blue", alpha=0.9)
    if metrics_baseline:
        axes[0].plot(smooth(metrics_baseline["fidelity"]), label="Hybrid 4Q (24 params)", color="darkorange", alpha=0.9)
    axes[0].plot(smooth(metrics_scaled["fidelity"]), label=f"Scaled Hybrid {agent_qubits}Q ({breakdown_scaled['quantum_variational_params']} params)", color="green", linewidth=2)
    axes[0].set_title("State Fidelity (Smoothed)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Fidelity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Reward
    if metrics_classic:
        axes[1].plot(smooth(metrics_classic["reward"]), label="Classical DQN", color="blue", alpha=0.9)
    if metrics_baseline:
        axes[1].plot(smooth(metrics_baseline["reward"]), label="Hybrid 4Q", color="darkorange", alpha=0.9)
    axes[1].plot(smooth(metrics_scaled["reward"]), label=f"Scaled Hybrid {agent_qubits}Q", color="green", linewidth=2)
    axes[1].set_title("Episode Reward (Smoothed)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # CNOT Count
    if metrics_classic:
        axes[2].plot(smooth(metrics_classic["cnot_count"]), label="Classical DQN", color="blue", alpha=0.9)
    if metrics_baseline:
        axes[2].plot(smooth(metrics_baseline["cnot_count"]), label="Hybrid 4Q", color="darkorange", alpha=0.9)
    axes[2].plot(smooth(metrics_scaled["cnot_count"]), label=f"Scaled Hybrid {agent_qubits}Q", color="green", linewidth=2)
    axes[2].set_title("CNOTs Used per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("CNOT Count")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "scaled_hybrid_teleport_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")
    print(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaled Hybrid Quantum RL Experiment")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--circuit", type=str, default="redundant_teleportation", help="Target circuit")
    parser.add_argument("--depth_discount", type=float, default=0.95, help="Depth discount factor")
    parser.add_argument("--noise", type=float, default=0.0, help="CNOT depolarizing noise")
    parser.add_argument("--agent_qubits", type=int, default=6, help="Controller qubits (default: 6)")
    parser.add_argument("--ansatz_layers", type=int, default=3, help="Ansatz layers (default: 3)")
    parser.add_argument("--no_reupload", action="store_true", help="Disable data re-uploading")
    parser.add_argument("--only_scaled", action="store_true", help="Only train scaled hybrid model")
    args = parser.parse_args()

    run_experiment(
        episodes=args.episodes,
        circuit_name=args.circuit,
        depth_discount=args.depth_discount,
        p_cnot=args.noise,
        agent_qubits=args.agent_qubits,
        ansatz_layers=args.ansatz_layers,
        use_data_reuploading=not args.no_reupload,
        only_scaled=args.only_scaled,
    )
