# Noise-Aware Quantum Reinforcement Learning for State Synthesis

## Executive Summary
In Noisy Intermediate-Scale Quantum (NISQ) devices, two-qubit entangling gates (such as CNOT) have error rates that are typically $10\times$ to $100\times$ higher than single-qubit gates. Standard compilation pipelines prioritize gate count or theoretical depth assuming unitary evolution, which can lead to severe fidelity degradation on real hardware.

This project investigates whether a **Reinforcement Learning agent** operating directly over a **noisy density matrix simulator** can discover **noise-resilient state preparation circuits** that outperform standard noiseless synthesis.

---

## System Architecture

```
q_rl/
├── envs/
│   ├── quantum_state_env.py    # Gymnasium environment with density matrix formalism
│   └── noise_models.py         # CNOT & 1-qubit depolarizing / dephasing Kraus operators
├── models/
│   ├── dqn_agent.py            # Double DQN with Dueling Architecture & Prioritized Buffer
│   └── hybrid_q_agent.py       # (Extension) Hybrid Quantum PQC-based policy
├── benchmarks/
│   ├── qiskit_baseline.py      # Standard Qiskit transpiler state prep baseline
│   └── evaluate_noise.py       # Sweep across varying CNOT error rates $p \in [0, 0.1]$
├── utils/
│   ├── quantum_metrics.py      # Uhlmann-Jozsa fidelity for mixed states, purity, trace dist
│   └── plotting.py             # Circuit rendering, fidelity vs episode, noise robustness curves
└── train.py                    # Main training pipeline with CLI flags
```

---

## Theoretical Formulation

### 1. State Space ($\mathcal{S}$)
For an $n$-qubit system, the quantum state is represented as a mixed state density matrix $\rho \in \mathbb{C}^{2^n \times 2^n}$:
- Properties: $\rho^\dagger = \rho$, $\operatorname{Tr}(\rho) = 1$, $\rho \ge 0$.
- Observation vector: Flattened real and imaginary components of $\rho$, or expectation values of Pauli basis strings $\langle \sigma_i \rangle = \operatorname{Tr}(\rho \sigma_i)$. For $n=2$ qubits, this is a 16-dimensional real vector.

### 2. Action Space ($\mathcal{A}$)
Discrete gate dictionary:
- **Single-qubit rotations**: $\{R_x(\pi/2), R_y(\pi/2), R_z(\pi/2), H, T\}$ on each qubit $i$.
- **Two-qubit gates**: $\text{CNOT}(i, j)$ for all directed pairs $(i, j)$ allowed by hardware coupling.
- **Terminal Action**: `STOP` (allows the agent to terminate early when high fidelity is reached).

### 3. Noise Modeling (Open Quantum Systems)
State evolution under quantum channel $\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$ where $\sum_k K_k^\dagger K_k = I$.
- **Single-qubit depolarizing noise**: Error rate $p_1 \approx 10^{-4}$ (or 0).
- **Two-qubit CNOT depolarizing noise**:
  $$\mathcal{E}_{\text{CNOT}}(\rho) = (1 - p_2) U_{\text{CNOT}} \rho U_{\text{CNOT}}^\dagger + \frac{p_2}{15} \sum_{i=1}^{15} P_i (U_{\text{CNOT}} \rho U_{\text{CNOT}}^\dagger) P_i$$
  where $P_i \in \{I, X, Y, Z\}^{\otimes 2} \setminus \{I \otimes I\}$.

### 4. Reward Formulation
At each step $t$:
$$R_t = \mathcal{F}(\rho_t, \sigma_{\text{target}}) - \mathcal{F}(\rho_{t-1}, \sigma_{\text{target}}) - c_{\text{gate}} - c_{\text{CNOT\_extra}}$$
Terminal reward:
$$R_{\text{terminal}} = +10 \cdot \mathbf{1}_{\mathcal{F} \ge \mathcal{F}_{\text{threshold}}}$$
where mixed state fidelity is $\mathcal{F}(\rho, |\psi\rangle\langle\psi|) = \langle\psi|\rho|\psi\rangle$.

---

## Experimental Workflow & Milestones

### Phase 1: Environment & Density Matrix Engine
- Implement `QuantumStateEnv` supporting density matrices $\rho$.
- Implement parameterized CNOT depolarizing noise channel.
- Target states: Bell state $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$, 3-qubit GHZ state $\frac{|000\rangle + |111\rangle}{\sqrt{2}}$, and 3-qubit W state.

### Phase 2: Double-DQN Agent Implementation
- Dueling DQN architecture with replay buffer, $\epsilon$-greedy exploration, and target network.
- State representation encoder (density matrix $\to$ feature vector).

### Phase 3: The Core Experiments
1. **Experiment A (Baseline):** Train agent in a **noiseless** environment ($p_2 = 0$). Evaluate on noisy test environment ($p_2 \in [0.01, 0.08]$).
2. **Experiment B (Noise-Aware Training):** Train agent **directly inside the noisy environment** ($p_2 = 0.03$).
3. **Experiment C (Comparison):**
   - Compare Fidelity & CNOT count:
     - Noiseless-trained Agent
     - Noise-aware Agent
     - Qiskit analytical baseline (e.g. `transpile(basis_gates=['rx', 'ry', 'rz', 'cx'])`)
   - Analyze whether the noise-aware agent finds circuits that minimize CNOT applications or replace CNOTs with combinations of single-qubit gates or alternative paths.

---

## Proposed Immediate Step
Begin **Phase 1**: build `envs/quantum_state_env.py` and `envs/noise_models.py` with unit tests verifying density matrix evolution, Kraus channel correctness, and fidelity calculations.
