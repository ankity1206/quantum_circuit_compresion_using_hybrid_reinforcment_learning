# Quantum-on-Quantum Reinforcement Learning (Q-on-Q RL) for Noise-Aware Circuit Synthesis

A hybrid quantum-classical reinforcement learning framework for compiling and optimizing quantum circuits under realistic open-system hardware noise.

A 4-qubit **Parameterized Quantum Circuit (PQC)** in PennyLane serves as the autonomous controller (policy network), learning to steer noisy quantum states into target entangled states using minimal gate depth.

---

## 🌟 Key Features

1. **Open-System Density Matrix Environment (`envs/quantum_state_env.py`)**:
   - Exact simulation of density matrices $\rho \in \mathbb{C}^{2^n \times 2^n}$ under Kraus depolarizing noise.
   - Observation space: 15-dimensional (2Q) or 63-dimensional (3Q) real Pauli expectation values $\langle P_i \rangle = \operatorname{Tr}(\rho P_i)$.
2. **Hybrid Quantum Policy & Scaled Architectures (`models/hybrid_q_agent.py`, `models/scaled_hybrid_q_agent.py`, `models/adaptive_hybrid_q_agent.py`)**:
   - 4-qubit base PennyLane variational quantum ansatz with adjoint differentiation (24 quantum parameters).
   - **Scaled 6-Qubit PQC with Data Re-uploading**: Expands the Fourier frequency spectrum and relieves the classical-to-quantum state bottleneck ($64 \to 6$) with 54 quantum parameters.
   - **Target-Aware Auto-Sizing & Progressive Growth (`adaptive_hybrid_q_agent.py`)**: Dynamically scales controller width $n_{\text{controller}} = \max(4, 2 \times n_{\text{target}})$ and expands capacity via identity-preserving weight transfer on plateau.
   - All-pairs pairwise correlator readout ($\binom{n}{2}$ correlators + $n$ single-qubit $Z$ expectations) feeding dueling value $V(s)$ and advantage $A(s, a)$ heads.
3. **Hardware-Realistic Cost Function**:
   - Differentiated physical gate penalties ($c_{1q} = 0.02$ for single-qubit rotations, $c_{2q} = 0.06$ for CNOTs).
   - Exponential depth discount factor $\gamma_{\text{depth}} = 0.95$ to strictly prune bloated circuits.
4. **Qiskit Integration (`utils/circuit_loader.py`)**:
   - Accepts OpenQASM 2.0 (`.qasm`) files or custom Qiskit circuits as inputs.
   - Renders ASCII circuit diagrams of both input and synthesized output circuits.

---

## 📁 Repository Structure

```
q_rl/
├── README.md                           # Project overview & reproduction guide
├── REPORT.md                           # Detailed empirical findings & theoretical analysis
├── PROJECT_PLAN.md                     # Roadmap and architectural blueprint
├── sample_circuit.qasm                 # Example input circuit
│
├── envs/
│   ├── quantum_state_env.py            # Gymnasium environment with density matrix physics
│   └── noise_models.py                 # 1Q and 2Q depolarizing Kraus channels
│
├── models/
│   ├── hybrid_q_agent.py               # PennyLane 4Q PQC + PyTorch Dueling DQN Agent (24 params)
│   ├── scaled_hybrid_q_agent.py        # Scaled 6Q PQC with Data Re-uploading & All-Pairs ZZ (54 params)
│   ├── adaptive_hybrid_q_agent.py      # Target-aware auto-sizing & progressive capacity expansion
│   └── dqn_agent.py                    # Classical Double Dueling DQN baseline (~16,000 params)
│
├── utils/
│   ├── circuit_loader.py               # QASM loader, state extractor, & circuit renderer
│   ├── quantum_metrics.py              # Fidelity, purity, and Pauli basis decomposition
│   └── plotting.py                     # Publication-quality plotting utilities
│
├── benchmarks/
│   ├── evaluate_noise.py               # CNOT noise robustness sweep (p in [0.0, 0.10])
│   ├── compare_teleportation.py        # 3-qubit teleportation depth discount benchmark
│   ├── experiment_scaled_hybrid.py     # Head-to-head benchmark: Classical vs 4Q vs Scaled 6Q
│   └── run_adaptive_hybrid.py          # Universal runner for arbitrary circuits / custom QASM
│
├── results/                            # All saved model weights, metrics, and plots
│   ├── hybrid_6q_scaled.pt             # Trained 6-qubit scaled hybrid model (100% fidelity)
│   ├── hybrid_depth_penalized.pt       # Trained 2-gate optimal hybrid model (4% noise)
│   ├── hybrid_noise_aware.pt           # Trained hybrid model (4% noise)
│   ├── dqn_with_discount_teleport.pt   # Trained classical model on 3Q teleportation
│   ├── scaled_hybrid_teleport_comparison.png # 3-way Classical vs 4Q vs 6Q comparison plot
│   ├── scaled_hybrid_teleport_results.json    # Quantitative metrics JSON
│   ├── noise_robustness.png            # Noise robustness curves
│   └── training_comparison.png         # Classical vs Hybrid training curves
│
├── train.py                            # Main training script
├── test_agent.py                       # Interactive model testing & circuit drawer
├── export_circuit_qasm.py              # Exports synthesized circuits to OpenQASM 2/3 & verifies
└── benchmark_user_circuit.py           # Evaluates custom Qiskit circuits under noise
```

---

## 🚀 How to Run & Reproduce

### 1. Test the Pre-trained Hybrid Quantum Agent:
```bash
# Test under 4% CNOT depolarizing noise
python3 test_agent.py --model results/hybrid_depth_penalized.pt --noise 0.04
```

### 2. Test with a Custom `.qasm` Circuit Input:
```bash
python3 test_agent.py --circuit sample_circuit.qasm --noise 0.04
```

### 3. Benchmark a Custom Circuit Under Noise:
```bash
python3 benchmark_user_circuit.py --circuit sample_circuit.qasm --noise 0.04
```

### 4. Train a New Hybrid Model:
```bash
python3 train.py \
  --agent_type hybrid \
  --circuit redundant_bell \
  --p_cnot 0.04 \
  --gate_penalty 0.02 \
  --cnot_penalty 0.06 \
  --depth_discount 0.95 \
  --target_fidelity 0.94 \
  --episodes 200 \
  --name my_hybrid_model
```

### 5. Run the 3-Qubit Teleportation Benchmark:
```bash
python3 benchmarks/compare_teleportation.py --episodes 200
```

### 6. Run Scaled 6Q Hybrid Head-to-Head Comparison:
```bash
# Full 3-way benchmark: Classical DQN vs. 4Q Hybrid vs. 6Q Scaled Hybrid
python3 benchmarks/experiment_scaled_hybrid.py --episodes 200

# Train only the 6-qubit scaled hybrid agent
python3 benchmarks/experiment_scaled_hybrid.py --only_scaled --episodes 200 --agent_qubits 6 --ansatz_layers 3
```

### 7. Universal Adaptive Hybrid Synthesis for Arbitrary Circuits:
```bash
# Auto-sizing (auto-allocates 4Q for 2-qubit targets, 6Q for 3-qubit targets)
python3 benchmarks/run_adaptive_hybrid.py --circuit bell --episodes 200
python3 benchmarks/run_adaptive_hybrid.py --circuit ghz3 --episodes 200
python3 benchmarks/run_adaptive_hybrid.py --circuit redundant_teleportation --episodes 200

# Custom user .qasm circuit
python3 benchmarks/run_adaptive_hybrid.py --circuit sample_circuit.qasm --episodes 200

# Progressive capacity expansion (starts at 4Q, expands to 6+ qubits on stagnation)
python3 benchmarks/run_adaptive_hybrid.py --circuit redundant_teleportation --progressive --patience 35 --episodes 250
```

### 8. Export Synthesized Circuits to OpenQASM (.qasm) & Verify:
```bash
# Export synthesized circuit from trained Scaled 6Q Hybrid agent & verify with noise
python3 export_circuit_qasm.py \
  --model results/hybrid_6q_scaled.pt \
  --circuit redundant_teleportation \
  --out results/synthesized_teleport_6q.qasm \
  --verify \
  --noise 0.04

# Export in OpenQASM 3.0 format
python3 export_circuit_qasm.py \
  --model results/hybrid_6q_scaled.pt \
  --circuit redundant_teleportation \
  --format qasm3 \
  --out results/synthesized_teleport_6q_v3.qasm

# Directly benchmark any exported QASM file with independent test harness
python3 benchmark_user_circuit.py \
  --circuit results/synthesized_teleport_6q.qasm \
  --noise 0.04
```

---

## 📊 Summary of Key Empirical Results

### 1. Circuit Compaction (Redundant Input -> Optimal Output):
- **Input Bloated Circuit (`redundant_bell`):** 4 gates, 3 CNOTs, Fidelity under noise = $0.9080$.
- **Synthesized Output Circuit:** **2 gates, 1 CNOT**, Fidelity under noise = **0.9680** (Physical ceiling!).

```text
Input Circuit:                    Synthesized Output Circuit:
     ┌───┐                             ┌───┐     
q_0: ┤ H ├──■────■────■──         q_0: ┤ H ├──■──
     └───┘┌─┴─┐┌─┴─┐┌─┴─┐              └───┘┌─┴─┐
q_1: ─────┤ X ├┤ X ├┤ X ├         q_1: ─────┤ X ├
          └───┘└───┘└───┘                   └───┘
```

### 2. Noise Robustness Across Error Rates ($p_2 \in [0.0, 0.10]$):

| CNOT Noise ($p_2$) | Noiseless-Trained Agent | Noise-Aware Agent | Advantage |
|:---:|:---:|:---:|:---:|
| **0.00** (Ideal) | 1.0000 | **1.0000** | 0.0000 |
| **0.02** | 0.4744 | **0.9840** | **+0.5096 (+107%)** |
| **0.04** (NISQ Average) | 0.4510 | **0.9680** | **+0.5170 (+114%)** |
| **0.08** | 0.4100 | **0.9360** | **+0.5260 (+128%)** |
| **0.10** | 0.3922 | **0.9200** | **+0.5278 (+134%)** |

### 3. 3-Qubit Scaling Benchmark: Classical vs. 4Q Baseline vs. Scaled 6Q Hybrid:

On the **3-qubit redundant teleportation target** ($|\mathcal{S}|=64$ dimensions, $|\mathcal{A}|=28$ actions):

| Metric | Classical DQN (MLP) | Baseline Hybrid (4Q, 2L) | **Scaled Hybrid (6Q, 3L + Re-uploading)** |
| :--- | :---: | :---: | :---: |
| **Target Achieved ($\ge 0.98$)?** | **YES** | **NO** | **YES (100%)** |
| **Final State Fidelity** | **1.0000** | $0.2500$ | **1.0000** |
| **Total Gates Synthesized** | **3 gates** | 12 gates | **3 gates** |
| **CNOTs Used** | **1 CNOT** | 0 CNOTs | **1 CNOT** |
| **CNOT Reduction** | **-75%** | N/A (Failed) | **-75%** |
| **Late-Training Success Rate** | 92.0% | 0.0% | **96.0% (Peak: 100%)** |
| **Variational Parameters** | ~16,000 (Classical) | 24 (Quantum) | **54 (Quantum) / 4,409 Total** |
| **Architecture** | Dense MLP | 4Q Single-Encode | **6Q Data Re-uploading + All-Pairs ZZ** |

```text
Classical DQN Output (Fidelity = 1.0000):
     ┌─────────┐
q_0: ┤ Ry(π/2) ├──■──
     └──┬───┬──┘  │  
q_1: ───┤ H ├─────┼──
        └───┘   ┌─┴─┐
q_2: ───────────┤ X ├
                └───┘

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

---

## 📜 Full Documentation
Detailed experimental logs, mathematical proofs, and theoretical analysis can be found in [`REPORT.md`](REPORT.md).
