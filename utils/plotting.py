import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_training_metrics(
    metrics_files: List[str],
    labels: List[str],
    save_path: str = "results/training_comparison.png",
    window: int = 15,
):
    """
    Plots smoothed fidelity, reward, and CNOT count across training episodes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for m_file, label in zip(metrics_files, labels):
        with open(m_file, "r") as f:
            data = json.load(f)

        episodes = data["episode"]
        fidelities = data["fidelity"]
        rewards = data["reward"]
        cnots = data["cnot_count"]

        # Moving average helper
        def moving_avg(arr, w):
            if len(arr) < w:
                return arr
            return np.convolve(arr, np.ones(w) / w, mode="valid")

        smooth_ep = episodes[window - 1 :] if len(episodes) >= window else episodes

        axes[0].plot(smooth_ep, moving_avg(fidelities, window), label=label, linewidth=2)
        axes[1].plot(smooth_ep, moving_avg(rewards, window), label=label, linewidth=2)
        axes[2].plot(smooth_ep, moving_avg(cnots, window), label=label, linewidth=2)

    axes[0].set_title("State Fidelity vs. Episode (Smoothed)", fontsize=13)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Target Fidelity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Episode Reward vs. Episode (Smoothed)", fontsize=13)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("CNOT Count vs. Episode (Smoothed)", fontsize=13)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("CNOTs in Synthesized Circuit")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training plot saved to: {save_path}")


def plot_noise_sweep(
    noise_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "results/noise_robustness.png",
):
    """
    Plots final state fidelity and purity vs CNOT error probability.
    noise_results format:
    {
       "Model A": {"p_errors": [...], "fidelities": [...], "purities": [...], "cnot_counts": [...]},
       "Model B": ...
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    markers = ["o", "s", "^", "v", "D"]

    for idx, (label, res) in enumerate(noise_results.items()):
        marker = markers[idx % len(markers)]
        axes[0].plot(
            res["p_errors"],
            res["fidelities"],
            label=f"{label} (avg CNOTs: {np.mean(res['cnot_counts']):.1f})",
            marker=marker,
            linewidth=2,
            markersize=7,
        )

        axes[1].plot(
            res["p_errors"],
            res["purities"],
            label=label,
            marker=marker,
            linewidth=2,
            markersize=7,
        )

    axes[0].set_title("Fidelity vs. CNOT Depolarizing Error Rate", fontsize=13)
    axes[0].set_xlabel("CNOT Error Rate $p_2$")
    axes[0].set_ylabel("Target State Fidelity $\\mathcal{F}$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("State Purity $\\mathrm{Tr}(\\rho^2)$ vs. CNOT Error Rate", fontsize=13)
    axes[1].set_xlabel("CNOT Error Rate $p_2$")
    axes[1].set_ylabel("Purity $\\gamma$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Noise sweep plot saved to: {save_path}")
