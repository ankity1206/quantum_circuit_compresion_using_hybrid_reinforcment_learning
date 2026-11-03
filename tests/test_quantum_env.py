import unittest
import numpy as np
from envs.quantum_state_env import QuantumStateEnv
from envs.noise_models import NoiseModel
from utils.quantum_metrics import purity, state_fidelity, get_bell_state


class TestQuantumEnvironment(unittest.TestCase):

    def test_noiseless_bell_state_synthesis(self):
        """Verify that H(0) followed by CNOT(0->1) synthesizes exact Bell state Phi+ with fidelity 1.0."""
        env = QuantumStateEnv(num_qubits=2, noise_model=NoiseModel(p_cnot=0.0))
        obs, info = env.reset()

        # Initial state is |00><00|; fidelity with (|00> + |11>)/sqrt(2) is |1/sqrt(2)|^2 = 0.5
        self.assertAlmostEqual(info["fidelity"], 0.5, places=5)
        self.assertAlmostEqual(info["purity"], 1.0, places=5)

        act_h0 = None
        act_cnot01 = None

        for idx, act in enumerate(env.actions_info):
            if act["name"] == "H(q0)":
                act_h0 = idx
            elif act["name"] == "CNOT(c0->t1)":
                act_cnot01 = idx

        self.assertIsNotNone(act_h0)
        self.assertIsNotNone(act_cnot01)

        # Step 1: H(0) creates (|00> + |10>)/sqrt(2)
        # Overlap with Phi+ is 1/2, so fidelity is (1/2)^2 = 0.25
        obs, r, term, trunc, info = env.step(act_h0)
        self.assertFalse(term)
        self.assertAlmostEqual(info["fidelity"], 0.25, places=5)
        self.assertAlmostEqual(info["purity"], 1.0, places=5)

        # Step 2: CNOT(0->1)
        obs, r, term, trunc, info = env.step(act_cnot01)
        self.assertAlmostEqual(info["fidelity"], 1.0, places=5)
        self.assertAlmostEqual(info["purity"], 1.0, places=5)
        self.assertTrue(term)  # Should terminate as fidelity >= target_fidelity (0.99)
        self.assertTrue(info["success"])

        # Check invariants
        trace_val = np.real(np.trace(env.rho))
        self.assertAlmostEqual(trace_val, 1.0, places=6)
        self.assertTrue(np.allclose(env.rho, env.rho.conj().T, atol=1e-6))

    def test_cnot_depolarizing_noise_reduces_purity(self):
        """Verify that CNOT depolarizing noise correctly lowers state purity and fidelity."""
        p_err = 0.05
        env_noisy = QuantumStateEnv(num_qubits=2, noise_model=NoiseModel(p_cnot=p_err))
        env_noisy.reset()

        act_h0 = [i for i, a in enumerate(env_noisy.actions_info) if a["name"] == "H(q0)"][0]
        act_cnot01 = [i for i, a in enumerate(env_noisy.actions_info) if a["name"] == "CNOT(c0->t1)"][0]

        env_noisy.step(act_h0)
        obs, r, term, trunc, info = env_noisy.step(act_cnot01)

        # Purity should be strictly less than 1.0 due to noise
        self.assertLess(info["purity"], 1.0)
        # Fidelity should be degraded from 1.0
        self.assertLess(info["fidelity"], 1.0)
        self.assertGreater(info["fidelity"], 0.90)

        # Check density matrix invariants under noise
        self.assertAlmostEqual(float(np.real(np.trace(env_noisy.rho))), 1.0, places=6)
        self.assertTrue(np.allclose(env_noisy.rho, env_noisy.rho.conj().T, atol=1e-6))
        eigenvalues = np.linalg.eigvalsh(env_noisy.rho)
        self.assertTrue(np.all(eigenvalues >= -1e-8))

    def test_observation_space_bounds(self):
        """Verify observation shape and range."""
        env = QuantumStateEnv(num_qubits=2)
        obs, info = env.reset()

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(np.all(obs >= -1.0 - 1e-5))
        self.assertTrue(np.all(obs <= 1.0 + 1e-5))


if __name__ == "__main__":
    unittest.main()
