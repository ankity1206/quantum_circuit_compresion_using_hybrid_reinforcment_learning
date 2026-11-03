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
from models.adaptive_hybrid_q_agent import (
    AdaptiveHybridDQNAgent,
    compute_optimal_agent_qubits,
    compute_optimal_ansatz_layers,
)
from utils.circuit_loader import (
    load_circuit_from_file_or_string,
    circuit_to_target_state,
    history_to_qiskit_circuit,
)


def run_adaptive_training(
    circuit_input: str = "redundant_teleportation",
    episodes: int = 200,
    depth_discount: float = 0.95,
    p_cnot: float = 0.0,
    target_fidelity: float = 0.98,
    auto_size: bool = True,
    progressive: bool = False,
    stagnation_patience: int = 35,
    save_dir: str = "results",
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load target circuit
    qc = load_circuit_from_file_or_string(circuit_input)
    num_qubits = qc.num_qubits
    target_state = circuit_to_target_state(qc)
    noise_model = NoiseModel(p_cnot=p_cnot)

    circuit_clean_name = os.path.splitext(os.path.basename(circuit_input))[0]
    model_name = f"adaptive_hybrid_{circuit_clean_name}"

    print("=" * 70)
    print("ADAPTIVE HYBRID QUANTUM-ON-QUANTUM REINFORCEMENT LEARNING")
    print(f"Target Plant        : {circuit_input} ({num_qubits} qubits)")
    print(f"Episodes            : {episodes}")
    print(f"Depth Discount      : {depth_discount:.2f}")
    print(f"CNOT Noise          : p = {p_cnot:.4f}")
    print(f"Auto-Sizing Mode    : {auto_size}")
    print(f"Progressive Growth  : {progressive}")
    print("=" * 70)
    print("Input Target Circuit Diagram:")
    print(qc.draw(output="text"))
    print("=" * 70)

    # 2. Setup Environment
    env = QuantumStateEnv(
        num_qubits=num_qubits,
        target_state=target_state,
        target_name=circuit_clean_name,
        target_fidelity=target_fidelity,
        noise_model=noise_model,
        gate_penalty=0.01,
        cnot_penalty=0.04,
        depth_discount=depth_discount,
        max_steps=12,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3. Setup Adaptive Agent
    agent = AdaptiveHybridDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_target_qubits=num_qubits,
        auto_size=auto_size,
        progressive_growth=progressive,
        stagnation_patience=stagnation_patience,
        target_fidelity=target_fidelity,
        learning_rate=2e-3,
        gamma=0.99,
        buffer_capacity=30000,
        batch_size=32,
        target_update_freq=25,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay_steps=max(600, int(episodes * 3.5)),
    )

    breakdown = agent.policy_net.get_parameter_breakdown()
    print("Initial Agent Architecture:")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    # 4. Training Loop
    metrics = {
        "episode": [],
        "reward": [],
        "fidelity": [],
        "purity": [],
        "cnot_count": [],
        "steps": [],
        "success": [],
        "history": [],
        "expansions": [],
    }

    start_time = time.time()
    agent_step = 0

    for ep in range(1, episodes + 1):
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

        # Check progressive growth trigger
        rolling_fid = np.mean(metrics["fidelity"][-15:])
        rolling_succ = np.mean(metrics["success"][-15:]) * 100
        expanded = agent.check_and_expand_if_stagnant(rolling_fid, rolling_succ)
        if expanded:
            metrics["expansions"].append(ep)

        if ep % 20 == 0 or ep == 1:
            avg_fid = np.mean(metrics["fidelity"][-20:])
            avg_cnots = np.mean(metrics["cnot_count"][-20:])
            succ_pct = np.mean(metrics["success"][-20:]) * 100
            print(
                f"Ep {ep:4d}/{episodes} | Rew: {ep_reward:6.2f} | "
                f"Fid (last 20): {avg_fid:.4f} | CNOTs: {avg_cnots:.1f} | "
                f"Success: {succ_pct:5.1f}% | Controller: {agent.agent_qubits}Q | Eps: {agent.epsilon:.3f}"
            )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {elapsed:.1f}s.")

    # 5. Deterministic Greedy Evaluation
    eval_obs, eval_info = env.reset()
    while True:
        eval_act = agent.select_action(eval_obs, evaluate=True)
        eval_obs, eval_r, eval_term, eval_trunc, eval_info = env.step(eval_act)
        if eval_term or eval_trunc:
            break

    out_qc = history_to_qiskit_circuit(eval_info["history"], num_qubits=num_qubits)
    print("=" * 70)
    print("Final Trained Greedy Output Circuit:")
    print(f"  Fidelity : {eval_info['fidelity']:.4f}")
    print(f"  Gates    : {eval_info['step']}")
    print(f"  CNOTs    : {eval_info['cnot_count']}")
    print(f"  Sequence : {' -> '.join(eval_info['history'])}")
    print("-" * 70)
    print(out_qc.draw(output="text"))
    print("=" * 70)

    # 6. Save Artifacts
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    agent.save(model_path)

    final_breakdown = agent.policy_net.get_parameter_breakdown()
    summary = {
        "target_circuit": circuit_input,
        "target_qubits": num_qubits,
        "fidelity": float(eval_info["fidelity"]),
        "total_gates": int(eval_info["step"]),
        "cnot_count": int(eval_info["cnot_count"]),
        "history": eval_info["history"],
        "training_time_seconds": float(elapsed),
        "final_controller_breakdown": final_breakdown,
        "expansions": metrics["expansions"],
    }

    summary_path = os.path.join(save_dir, f"{model_name}_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 7. Generate Training Curve Plot
    def smooth(arr, w=15):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Fidelity
    axes[0].plot(smooth(metrics["fidelity"]), color="teal", linewidth=2, label="Adaptive Hybrid")
    axes[0].axhline(target_fidelity, color="red", linestyle="--", label=f"Target ({target_fidelity})")
    for exp_ep in metrics["expansions"]:
        axes[0].axvline(exp_ep, color="purple", linestyle=":", label="Qubit Expansion" if exp_ep == metrics["expansions"][0] else "")
    axes[0].set_title(f"State Fidelity ({circuit_clean_name})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Fidelity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Reward
    axes[1].plot(smooth(metrics["reward"]), color="royalblue", linewidth=2)
    for exp_ep in metrics["expansions"]:
        axes[1].axvline(exp_ep, color="purple", linestyle=":")
    axes[1].set_title("Episode Reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)

    # CNOTs
    axes[2].plot(smooth(metrics["cnot_count"]), color="darkorange", linewidth=2)
    for exp_ep in metrics["expansions"]:
        axes[2].axvline(exp_ep, color="purple", linestyle=":")
    axes[2].set_title("CNOT Gates per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("CNOTs")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name}_training.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Model saved to: {model_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Plot saved to: {plot_path}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Hybrid Quantum RL for Arbitrary Circuits")
    parser.add_argument("--circuit", type=str, default="redundant_teleportation", help="Target circuit (.qasm or preset)")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--depth_discount", type=float, default=0.95, help="Depth discount factor")
    parser.add_argument("--noise", type=float, default=0.0, help="CNOT depolarizing noise rate")
    parser.add_argument("--target_fidelity", type=float, default=0.98, help="Target fidelity threshold")
    parser.add_argument("--progressive", action="store_true", help="Enable progressive capacity expansion on stagnation")
    parser.add_argument("--patience", type=int, default=35, help="Stagnation patience before expanding")
    args = parser.parse_args()

    run_adaptive_training(
        circuit_input=args.circuit,
        episodes=args.episodes,
        depth_discount=args.depth_discount,
        p_cnot=args.noise,
        target_fidelity=args.target_fidelity,
        auto_size=not args.progressive,
        progressive=args.progressive,
        stagnation_patience=args.patience,
    )
