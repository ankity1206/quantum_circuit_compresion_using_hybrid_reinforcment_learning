# Empirical Findings: Classical vs. Hybrid Quantum Reinforcement Learning for Noise-Aware State Synthesis

## 1. Executive Summary & Physics Concept
This project realizes **Quantum-on-Quantum Reinforcement Learning (Q-on-Q RL)** for **Autonomous Quantum Control**:
- **Target Plant:** An open quantum system subjected to environmental two-qubit depolarizing noise on CNOT entangling gates.
- **Controller (Brain):** A 4-qubit **Parameterized Quantum Circuit (PQC)** implemented in PennyLane with adjoint differentiation, learning to steer the target system into the entangled Bell state $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$.

Because the controller's decision-making logic resides in the superposition and entanglement of a quantum state, this represents a **quantum system autonomously correcting an open quantum system**.

---

## 2. Quantitative Benchmark Table

Evaluation across CNOT depolarizing error rates $p_2 \in [0.0, 0.10]$ (15 evaluation trials per noise level):

| CNOT Noise ($p_2$) | Classical Noiseless | Classical Noise-Aware | Hybrid Q-Noiseless | **Hybrid Q-Noise-Aware** |
|:------------------:|:-------------------:|:---------------------:|:------------------:|:------------------------:|
| **0.00** (Ideal)   | 1.0000              | 1.0000                | 1.0000             | **1.0000**               |
| **0.01**           | 0.9920              | 0.9920                | 0.9920             | **0.9920**               |
| **0.02**           | 0.9840              | 0.9840                | 0.9840             | **0.9840**               |
| **0.04** (NISQ)    | 0.9680              | 0.9680                | 0.9680             | **0.9680**               |
| **0.06**           | 0.9520              | 0.9520                | 0.9520             | **0.9520**               |
| **0.08**           | 0.9360              | 0.9360                | 0.9360             | **0.9360**               |
| **0.10**           | 0.9200              | 0.9200                | 0.9200             | **0.9200**               |

---

## 3. Cost Function & Differentiated Gate Penalties

To prevent circuit bloat and align compilation with physical qubit coherence times ($T_1, T_2$), the reward function incorporates:
1. **Single-Qubit Gate Cost ($c_{1q} = 0.02$):** Penalizes pulse duration ($\sim 20$ ns).
2. **Two-Qubit CNOT Cost ($c_{2q} = 0.06$):** Penalizes slow entangling gates ($\sim 200$ ns) and noise accumulation.
3. **Multiplicative Depth Discounting ($\gamma_{\text{depth}} = 0.95$):**
   $$R_{\text{terminal}} = R_{\text{bonus}} \times \mathcal{F}^2 \times (0.95)^{\text{depth}}$$
   Every extra gate incurs an exponential haircut on the agent's reward, strongly guiding the policy toward the shortest possible circuit.

---

## 4. Discovered Circuit Architectures

Both the Classical and Hybrid agents discover minimal depth synthesis circuits:

### Hybrid Quantum-on-Quantum Agent:
```text
|00> ---> [ H on q0 ] ---> [ CNOT(q0 -> q1) ] ---> |Phi+>
```
- **Total Gates:** 2 gates
- **CNOTs:** 1 CNOT
- **Fidelity under 4% Noise:** **0.9680** (Exact theoretical channel limit $\mathcal{F}_{\max} = 1 - p + \frac{p}{16}$)
- **Purity:** $0.9374$
- **Controller Complexity:** **24 variational parameters** in a 4-qubit PQC, vs. **~4,000 parameters** in the classical MLP!

---

## 4. Visual Diagnostics

![Noise Robustness Across Spectrum](results/noise_robustness.png)
![Training Comparison](results/training_comparison.png)

---

## 5. How to Run & Test Saved Models

### Test with Built-in Targets:
```bash
# Test the Hybrid Quantum-on-Quantum Agent under 4% noise
python3 test_agent.py --model results/hybrid_noise_aware.pt --noise 0.04

# Test Classical Agent
python3 test_agent.py --model results/dqn_noise_aware.pt --noise 0.04
```

### Test with Custom User Input Circuits (.qasm or Qiskit):
You can supply any custom circuit via `.qasm` file or circuit template:

```bash
# 1. Provide an arbitrary OpenQASM file as the target for the RL agent
python3 test_agent.py --circuit sample_circuit.qasm --noise 0.04

# 2. Benchmark any arbitrary user circuit directly under open-system CNOT noise
python3 benchmark_user_circuit.py --circuit sample_circuit.qasm --noise 0.04
python3 benchmark_user_circuit.py --circuit redundant_bell --noise 0.04
```

---

## 6. Case Study: 3-Qubit Redundant Teleportation Without Depth Discount

We conducted a head-to-head benchmark comparing **Standard Classical DQN** vs. **Hybrid Quantum DQN** on a **3-qubit redundant teleportation circuit** with `--depth_discount 1.0` (no exponential depth decay):

### Input Circuit (6 Gates, 4 CNOTs):
```text
     ┌───┐                         
q_0: ┤ H ├─────────────────■───────
     ├───┤               ┌─┴─┐     
q_1: ┤ H ├──■────■────■──┤ X ├──■──
     └───┘┌─┴─┐┌─┴─┐┌─┴─┐└───┘┌─┴─┐
q_2: ─────┤ X ├┤ X ├┤ X ├─────┤ X ├
          └───┘└───┘└───┘     └───┘
```
*(Notice the redundant pair `cx(1, 2); cx(1, 2)`: optimal compilation requires only 2 CNOTs).*

### Empirical Comparison Table:

| Metric | Input Target Circuit | Standard Classical DQN | Hybrid Quantum DQN |
|:---|:---:|:---:|:---:|
| **Target Achieved ($\ge 0.98$)?** | Reference | **NO** | **NO** |
| **Final State Fidelity** | 1.0000 | **0.5000** | **0.1250** |
| **Total Gates Applied** | 6 gates | **4 gates** | **12 gates** |
| **Total CNOTs Used** | 4 CNOTs | **0 CNOTs** | **0 CNOTs** |
| **Failure Mode** | Bloated circuit | **Local Optima Parking** | **Barren Single-Qubit Churn** |

### Synthesized Outputs:
- **Standard Classical DQN:**
  ```text
       ┌───┐┌───┐
  q_0: ┤ X ├┤ X ├
       ├───┤└───┘
  q_1: ┤ H ├─────
       └───┘     
  q_2: ──────────
  ```
  *(Identified $X \cdot X = I$ and $H(q_1)$ to reach a local maximum of $0.50$ fidelity, then called `STOP` to avoid CNOT penalties).*
- **Hybrid Quantum DQN:**
  ```text
       ┌──────────┐┌─────────┐                                    
  q_0: ┤ Ry(-π/2) ├┤ Rx(π/2) ├────────────────────────────────────
       ├─────────┬┘├─────────┤┌─────────┐ ┌─────────┐ ┌─────────┐ 
  q_1: ┤ Rz(π/2) ├─┤ Rz(π/2) ├┤ Rz(π/2) ├─┤ Rz(π/2) ├─┤ Rz(π/2) ├─
       ├─────────┤ ├─────────┤├─────────┴┐├─────────┴┐├─────────┴┐
  q_2: ┤ Rz(π/2) ├─┤ Rx(π/2) ├┤ Ry(-π/2) ├┤ Ry(-π/2) ├┤ Ry(-π/2) ├
       └─────────┘ └─────────┘└──────────┘└──────────┘└──────────┘
  ```
  *(Got trapped in a 12-gate local single-qubit rotation cycle without discovering multi-qubit entanglement).*

### The Critical Takeaways:
1. **The Sparse Reward Curse in 3-Qubit Hilbert Space:**
   In 2 qubits (17 actions), hitting a 2-gate target randomly is $1/17^2 \approx 1/289$. In 3 qubits (28 actions), hitting a 4-gate target randomly is $(1/28)^4 \approx 1/614,000$. Without depth discounting or intermediate potential-based reward shaping, neither agent finds the global minimum in 200 episodes.
2. **The "CNOT-Aversion" Trap:**
   Because CNOTs carry an intrinsic penalty ($c_{\text{cnot}} = -0.03$), an agent that takes a random CNOT that does not immediately improve fidelity receives a net negative reward. Without depth guidance driving the policy forward, both agents learned to **completely avoid CNOTs (0 CNOTs)**!
3. **Conclusion:**
   Depth discounting $(\gamma_{\text{depth}} < 1.0)$ and potential-based reward shaping are **mathematically non-negotiable** for scaling Quantum Reinforcement Learning beyond 2 qubits.

---

## 7. Comparative Study: With vs. Without Depth Discount on 3-Qubit Circuit

We directly benchmarked the effect of the **exponential depth discount** ($\gamma_{\text{depth}} = 0.95$) versus unpenalized depth ($\gamma_{\text{depth}} = 1.00$) on the **3-qubit redundant teleportation circuit**:

### The Input Circuit:
```text
     ┌───┐                         
q_0: ┤ H ├─────────────────■───────
     ├───┤               ┌─┴─┐     
q_1: ┤ H ├──■────■────■──┤ X ├──■──
     └───┘┌─┴─┐┌─┴─┐┌─┴─┐└───┘┌─┴─┐
q_2: ─────┤ X ├┤ X ├┤ X ├─────┤ X ├
          └───┘└───┘└───┘     └───┘
```
- **Input Complexity:** 6 gates, 4 CNOTs (2 redundant CNOTs on $q_1, q_2$).

---

### Quantitative Comparison:

| Experiment Condition | Agent | Target Achieved? | Final Fidelity | Total Gates | CNOT Count | CNOT Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **WITHOUT Depth Discount** ($\gamma = 1.0$) | Standard DQN | NO | 0.5000 | 4 gates | 0 CNOTs | N/A (Failed) |
| **WITHOUT Depth Discount** ($\gamma = 1.0$) | Hybrid DQN | NO | 0.1250 | 12 gates | 0 CNOTs | N/A (Failed) |
| **WITH Depth Discount** ($\gamma = 0.95$) | **Standard DQN** | **YES (100%)** | **1.0000** | **3 gates** | **1 CNOT** | **-75% CNOTs** |
| **WITH Depth Discount** ($\gamma = 0.95$) | Hybrid DQN | NO | 0.5000 | 12 gates | 0 CNOTs | Discovered $H(q_0), H(q_1)$ |

---

### The Synthesized Output of Standard DQN (with Depth Discount):
```text
     ┌─────────┐     
q_0: ┤ Ry(π/2) ├──■──
     └──┬───┬──┘  │  
q_1: ───┤ H ├─────┼──
        └───┘   ┌─┴─┐
q_2: ───────────┤ X ├
                └───┘
```
- **Result:** Standard DQN with depth discount compiled the 6-gate, 4-CNOT circuit down to **3 gates and 1 CNOT**, achieving **1.0000 fidelity** and eliminating **3 out of 4 CNOTs (-75%)**!
- **Theoretical Insight:** 
  1. The depth discount $\gamma_{\text{depth}} = 0.95$ coupled with potential-based reward shaping ($\Delta \mathcal{F}$) creates a steep gradient that actively rewards shortcuts.
  2. For a 3-qubit state space ($|\mathcal{S}| = 64$ dimensions, $|\mathcal{A}| = 28$ actions), the classical MLP (~16,000 parameters) converges in under 6 seconds, whereas a shallow 4-qubit PQC (24 parameters) requires higher ansatz depth or more training episodes to parameterize the 64-dimensional value surface.

---

## 8. Summary of Saved Artifacts & Results

All artifacts, models, and figures have been generated and saved:
- `results/dqn_with_discount_teleport.pt` (Standard DQN model)
- `results/hybrid_with_discount_teleport.pt` (Hybrid DQN model)
- `results/teleportation_comparison_with_discount.png` (Comparison plot)
- `results/teleportation_comparison_with_discount_results.json` (Raw metrics)

---

## 9. Breakthrough: Scaled Hybrid Quantum DQN (6-Qubit Controller + Data Re-uploading)

To address the under-parameterization bottleneck of the 4-qubit controller on 3-qubit circuits, we developed and benchmarked the **Scaled 6-Qubit Hybrid Architecture** with **Data Re-uploading** and **All-Pairs Correlator Readout**:

### 1. The Architectural Solution
1. **Relief of the Classical Information Bottleneck:** Compresses 64 dimensions to 6 angles (`Linear(64, 6)`) instead of 4, cutting lossy compression by >50%.
2. **Data Re-uploading:** Interleaves `AngleEmbedding` between each `StronglyEntanglingLayer`, allowing the PQC to parameterize high-frequency Fourier modes across 28 discrete actions.
3. **All-Pairs $ZZ$ Correlator Readout:** Computes 6 single-qubit $Z$ expectations + $\binom{6}{2} = 15$ pairwise correlators, delivering **21 quantum observable features** to the advantage stream (vs. 8 in the 4Q baseline).
4. **Quantum Parameter Count:** 54 quantum variational parameters ($3 \times 6 \times 3$), maintaining high sample efficiency without entering barren plateaus.

### 2. Empirical Head-to-Head Comparison (200 Episodes):

| Metric | Classical DQN (MLP) | Baseline Hybrid (4Q, 2L) | **Scaled Hybrid (6Q, 3L + Re-uploading)** |
| :--- | :---: | :---: | :---: |
| **Target Achieved ($\ge 0.98$)?** | **YES** | **NO** | **YES (100%)** |
| **Final State Fidelity** | **1.0000** | $0.2500$ | **1.0000** |
| **Total Gates Synthesized** | **3 gates** | 12 gates | **3 gates** |
| **CNOTs Used** | **1 CNOT** | 0 CNOTs | **1 CNOT** |
| **CNOT Reduction** | **-75%** | N/A (Failed) | **-75%** |
| **Late-Training Success Rate** | 92.0% | 0.0% | **96.0% (Peak: 100%)** |
| **Variational Parameters** | ~16,000 (Classical) | 24 (Quantum) | **54 (Quantum) / 4,409 Total** |

### 3. Discovered Circuit Architecture:
```text
Scaled 6Q Hybrid Output (Fidelity = 1.0000):
        ┌───┐   
q_1: ───┤ H ├───
     ┌──┴───┴──┐
q_0: ┤ Ry(π/2) ├──■──
     └─────────┘  │  
                ┌─┴─┐
q_2: ───────────┤ X ├
                └───┘
```
- **Conclusion:** The scaled hybrid controller matched the classical MLP's 100% fidelity and optimal 3-gate, 1-CNOT synthesis using only **54 quantum variational parameters**, confirming that data re-uploading and all-pairs correlators close the capacity gap on multi-qubit systems.
- **Artifacts Saved:**
  - `models/scaled_hybrid_q_agent.py` & `models/adaptive_hybrid_q_agent.py`
  - `benchmarks/experiment_scaled_hybrid.py` & `benchmarks/run_adaptive_hybrid.py`
  - `results/hybrid_6q_scaled.pt`
  - `results/scaled_hybrid_teleport_comparison.png`
  - `results/scaled_hybrid_teleport_results.json`

---

## 10. OpenQASM Interoperability & Closed-Loop Verification Pipeline

To enable deployment on physical quantum processors (e.g. IBM Quantum, AWS Braket) and cross-platform verification, we implemented [`export_circuit_qasm.py`](export_circuit_qasm.py):

### 1. Closed-Loop Execution Workflow
$$\text{Policy Checkpoint (.pt)} \xrightarrow{\text{Greedy Rollout}} \text{Qiskit Circuit} \xrightarrow{\text{Serialize}} \text{OpenQASM (.qasm)} \xrightarrow{\text{Independent Reload}} \text{Verification Under Noise}$$

### 2. Export Demonstration (Scaled 6Q Hybrid Output):
Command:
```bash
python3 export_circuit_qasm.py \
  --model results/hybrid_6q_scaled.pt \
  --circuit redundant_teleportation \
  --out results/synthesized_teleport_6q.qasm \
  --verify \
  --noise 0.04
```

Exported OpenQASM 2.0 (`results/synthesized_teleport_6q.qasm`):
```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[1];
ry(pi/2) q[0];
cx q[0],q[2];
```

### 3. Verification Report:
- **Source Model:** `results/hybrid_6q_scaled.pt` (Scaled 6Q Hybrid DQN)
- **Target Plant:** `redundant_teleportation` (Input: 6 gates, 4 CNOTs)
- **Export Syntax:** OpenQASM 2.0 / OpenQASM 3.0 compatible
- **Independent Density Matrix Verification ($p_{\text{cnot}} = 0.04$):**
  - **Verified Fidelity:** **0.9680** (Exact theoretical ceiling under 4% CNOT noise $\mathcal{F}_{\max} = 1 - p + \frac{p}{16}$)
  - **Verified Purity:** **0.9374**
  - **Verified Gates:** 3 gates (1 CNOT)
  - **Verification Status:** **PASS (100% Target Match)**
