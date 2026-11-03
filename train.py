import argparse
import os
import time
import json
import numpy as np
import torch

from envs.quantum_state_env import QuantumStateEnv
from envs.noise_models import NoiseModel
from models.dqn_agent import DQNAgent
from models.hybrid_q_agent import HybridDQNAgent
from utils.circuit_loader import load_circuit_from_file_or_string, circuit_to_target_state


def train(
    num_episodes: int = 400,
    num_qubits: int = 2,
    target_name: str = "bell_phi_plus",
    circuit_input: str = None,
    target_fidelity: float = None,
    agent_type: str = "classical",
    p_cnot: float = 0.0,
    gate_penalty: float = 0.02,
    cnot_penalty: float = 0.06,
    depth_discount: float = 0.95,
    max_steps: int = 12,
    seed: int = 42,
    save_dir: str = "results",
    model_name: str = "dqn_model",
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    target_state = None
    if circuit_input is not None:
        qc = load_circuit_from_file_or_string(circuit_input)
        num_qubits = qc.num_qubits
        target_state = circuit_to_target_state(qc)
        target_name = f"CustomCircuit({circuit_input})"
        print("=" * 65)
        print("Input Target Circuit:")
        print(qc.draw(output="text"))

    noise_model = NoiseModel(p_cnot=p_cnot)
    # When CNOT noise is present, physical channel capacity limits fidelity to ~1 - p
    if target_fidelity is None:
        target_fid = 0.95 if p_cnot > 0.0 else 0.99
    else:
        target_fid = target_fidelity

    env = QuantumStateEnv(
        num_qubits=num_qubits,
        target_state=target_state,
        target_name=target_name,
        target_fidelity=target_fid,
        noise_model=noise_model,
        gate_penalty=gate_penalty,
        cnot_penalty=cnot_penalty,
        depth_discount=depth_discount,
        max_steps=max_steps,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if agent_type == "hybrid":
        agent = HybridDQNAgent(
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
            epsilon_decay_steps=max(600, int(num_episodes * 3.5)),
        )
    else:
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=5e-4,
            gamma=0.99,
            buffer_capacity=30000,
            batch_size=64,
            target_update_freq=20,
            epsilon_start=1.0,
            epsilon_end=0.03,
            epsilon_decay_steps=max(500, int(num_episodes * 4)),
        )

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

    print("=" * 65)
    print(f"Training DQN Agent: {model_name}")
    print(f"Target: {target_name} ({num_qubits} qubits) | CNOT Noise p={p_cnot}")
    print(f"Obs Dim: {state_dim} | Actions: {action_dim} | Episodes: {num_episodes}")
    print("=" * 65)

    start_time = time.time()

    agent_step = 0
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        losses = []

        while True:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            agent_step += 1

            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            if agent_step % 2 == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

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
                f"Ep {ep:4d}/{num_episodes} | Rew: {ep_reward:6.2f} | "
                f"Fid (last 25): {avg_fid:.4f} | CNOTs: {avg_cnots:.1f} | "
                f"Success: {success_rate:5.1f}% | Eps: {agent.epsilon:.3f}"
            )

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"Training completed in {elapsed:.1f}s.")

    # Run deterministic evaluation rollout with greedy policy (no exploration noise)
    eval_obs, eval_info = env.reset()
    while True:
        eval_act = agent.select_action(eval_obs, evaluate=True)
        eval_obs, eval_r, eval_term, eval_trunc, eval_info = env.step(eval_act)
        if eval_term or eval_trunc:
            break

    from utils.circuit_loader import history_to_qiskit_circuit
    out_qc = history_to_qiskit_circuit(eval_info["history"], num_qubits=num_qubits)
    print("=" * 65)
    print("Final Trained Greedy Output Circuit:")
    print(f"  Fidelity: {eval_info['fidelity']:.4f} | Gates: {eval_info['step']} | CNOTs: {eval_info['cnot_count']}")
    print(f"  Sequence: {' -> '.join(eval_info['history'])}")
    print("-" * 65)
    print(out_qc.draw(output="text"))
    print("=" * 65)

    # Save model and metrics
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    agent.save(model_path)

    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 65)

    return agent, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--circuit", type=str, default=None, help="Input circuit path (.qasm) or template name")
    parser.add_argument(
        "--target_fidelity",
        "--threshold",
        type=float,
        default=None,
        help="Target fidelity threshold (defaults to physical capacity ~0.95 under noise)",
    )
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--target", type=str, default="bell_phi_plus")
    parser.add_argument("--agent_type", type=str, default="classical", choices=["classical", "hybrid"])
    parser.add_argument("--p_cnot", type=float, default=0.0)
    parser.add_argument("--gate_penalty", type=float, default=0.02)
    parser.add_argument("--cnot_penalty", type=float, default=0.06)
    parser.add_argument("--depth_discount", type=float, default=0.95)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="dqn_noiseless")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        num_qubits=args.qubits,
        target_name=args.target,
        circuit_input=args.circuit,
        target_fidelity=args.target_fidelity,
        agent_type=args.agent_type,
        p_cnot=args.p_cnot,
        gate_penalty=args.gate_penalty,
        cnot_penalty=args.cnot_penalty,
        depth_discount=args.depth_discount,
        max_steps=args.max_steps,
        seed=args.seed,
        model_name=args.name,
    )
