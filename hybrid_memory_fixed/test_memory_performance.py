"""Independent checks of the storage-performance calculation.

Run from the repository root:
python -m unittest discover -s hybrid_memory_fixed -p 'test_memory_performance.py'
"""
import unittest
from unittest.mock import patch

import numpy as np
import qutip as qt

import backend as b


class MemoryPerformanceTests(unittest.TestCase):
    def test_transfer_curves_score_failed_transfer_in_both_directions(self):
        # Identity evolution leaves the source intact. Only logical zero
        # matches fully; + has overlap 1/4, rather than a false perfect return.
        def unchanged_write(alpha, beta, **kwargs):
            self.assertFalse(kwargs["include_storage"])
            state = qt.tensor((alpha*b.g + beta*b.e).unit(), qt.basis(b.NC, 0), b.h0)
            return np.array([0., 1.]), [state, state]

        def unchanged_read(initial, **kwargs):
            # Read inputs must have an empty transmon and cavity.
            self.assertAlmostEqual(qt.expect(b.Pe, initial), 0)
            self.assertAlmostEqual(qt.expect(b.n, initial), 0)
            return np.array([0., 1.]), [initial, initial], [], None, None

        b.transfer_fidelity_data.cache_clear()
        self.addCleanup(b.transfer_fidelity_data.cache_clear)
        with patch.object(b, "simulate", side_effect=unchanged_write), \
             patch.object(b, "stored_retrieval_initial", side_effect=b.retrieval_initial), \
             patch.object(b, "evolve_read", side_effect=unchanged_read), \
             patch.object(b, "calibrated_phase", return_value=0), \
             patch.object(b, "read_phase", return_value=0):
            for mode in ("write", "read"):
                data = b.transfer_fidelity_data(mode)
                np.testing.assert_allclose(data[:, 1:], [[1., 0., .25]]*2, atol=1e-12)

    def test_read_fidelity_includes_storage_dephasing(self):
        # A 100 us hold with Tphi=10 us destroys the phase of a balanced
        # hyperfine qubit while preserving both logical populations.
        b.parked_storage_solver.cache_clear()
        self.addCleanup(b.parked_storage_solver.cache_clear)
        collapse = [np.sqrt(1 / (2 * 10.)) * b.Zh]
        with patch.object(b, "C_OPS", collapse), patch.object(b, "STORAGE", 100.):
            held = b.stored_retrieval_initial(1, 1)
            self.assertAlmostEqual(qt.expect(b.Ph0, held), .5, delta=1e-8)
            self.assertAlmostEqual(qt.expect(b.Ph1, held), .5, delta=1e-8)
            self.assertAlmostEqual(2*abs(held.ptrace(2)[0, 1]), np.exp(-10), delta=1e-8)
            _, _, fidelities = b.basis_probabilities("read")
            self.assertGreater(fidelities[0], .99)
            self.assertGreater(fidelities[1], .99)
            self.assertAlmostEqual(fidelities[2], .5, delta=.01)

    def test_default_figures_use_independent_retrieval(self):
        with patch.object(b, "main") as write, \
             patch.object(b, "retrieval_main") as read:
            b.all_figures()
        write.assert_called_once_with(save=False)
        read.assert_called_once_with(save=False)

    def test_write_output_scores_the_state_after_storage(self):
        b.parked_storage_solver.cache_clear()
        self.addCleanup(b.parked_storage_solver.cache_clear)
        collapse = [np.sqrt(1 / (2 * 100.)) * b.Zh]

        def perfect_write(alpha, beta, **kwargs):
            state = b.retrieval_initial(alpha, beta)
            return np.array([0.]), [state]

        with patch.object(b, "C_OPS", collapse), patch.object(b, "STORAGE", 500.), \
             patch.object(b, "simulate", side_effect=perfect_write):
            _, _, immediate = b.basis_probabilities("write", phase=0, storage=False)
            _, _, held = b.basis_probabilities("write", phase=0, storage=True)
        np.testing.assert_allclose(immediate, [1, 1, 1], atol=1e-8)
        np.testing.assert_allclose(held, [1, 1, (1 + np.exp(-5))/2], atol=1e-8)

    def test_default_excitation_survives_write_storage_and_read(self):
        # Calibrated rounded pulses must succeed with spectator coupling.
        # This catches the former default's ~15% excited-input retrieval.
        _, written = b.simulate(0, 1, noisy=False, include_storage=False)
        self.assertGreater(qt.expect(b.atomic_storage_effect(b.e, 0), written[-1]), .995)
        target = qt.tensor(b.e, qt.basis(b.NC, 0), b.h0)
        for hold in (0., b.STORAGE, 2*b.STORAGE):
            with self.subTest(hold=hold):
                stored = (-1j*b.H(b.MAP[1])*hold).expm()*written[-1]
                _, read, _, _, _ = b.evolve_read(stored, noisy=False)
                self.assertGreater(qt.expect(target.proj(), read[-1]), .995)

    def test_unchanged_transmon_is_not_successful_atomic_storage(self):
        inputs = (b.g, b.e, (b.g+b.e).unit(), (b.g-b.e).unit(),
                  (b.g+1j*b.e).unit(), (b.g-1j*b.e).unit())
        atomic = []
        for target in inputs:
            bypass = qt.tensor(target, qt.basis(b.NC, 0), b.h0)
            returned = b.read_metrics([bypass], target, 0)[2][0]
            self.assertAlmostEqual(returned, 1)
            atomic.append(qt.expect(b.atomic_storage_effect(target, 0), bypass))
        self.assertAlmostEqual(atomic[1], 0)
        self.assertAlmostEqual(np.mean(atomic), 1/3)

    def test_atomic_target_checks_phase_and_counts_loss_without_normalizing(self):
        target = (b.g + 1j*b.e).unit()
        phase = .63
        desired = qt.tensor(b.g, qt.basis(b.NC, 0), (b.h0 + 1j*b.h1).unit())
        rotation = qt.tensor(b.IT, b.IC, b.memory_rotation(phase))
        written = rotation.dag() * desired
        effect = b.atomic_storage_effect(target, phase)
        self.assertAlmostEqual(qt.expect(effect, written), 1)
        wrong_phase = qt.tensor(b.g, qt.basis(b.NC, 0), (b.h0 - 1j*b.h1).unit())
        self.assertAlmostEqual(qt.expect(b.atomic_storage_effect(target, 0), wrong_phase), 0)
        lost = qt.tensor(b.g, qt.basis(b.NC, 0), b.lost)
        mixed = .7 * written.proj() + .3 * lost.proj()
        self.assertAlmostEqual(qt.expect(effect, mixed), .7)

    def test_adjoint_readout_matches_forward_with_phase_and_loss(self):
        target = (b.g + 1j * b.e).unit()
        initial = (.8 * b.retrieval_initial(1, 1j).proj()
                   + .2 * qt.tensor(b.g, qt.basis(b.NC, 0), b.lost).proj())
        phase = .37
        for noisy in (False, True):
            with self.subTest(noisy=noisy):
                effect = b.retrieval_effect(target, phase, noisy=noisy)
                _, states, _, _, _ = b.evolve_read(initial, noisy=noisy)
                forward = b.read_metrics(states[-1:], target, phase)[2][-1]
                self.assertAlmostEqual(qt.expect(effect, initial), forward, delta=2e-6)

    def test_constant_storage_matches_time_dependent_model(self):
        initial = qt.tensor((b.g + b.e).unit(), qt.basis(b.NC, 0),
                            (b.h0 + b.h1).unit()).proj()
        hold = .003
        exact = qt.mesolve(b.H(b.MAP[1]), initial, [0, hold], b.C_OPS,
                           options={"method": "diag", "normalize_output": False}).states[-1]
        reference = qt.mesolve(b.H, initial, [b.MAP[1], b.MAP[1] + hold], b.C_OPS,
                               options={"method": "dop853", "atol": 1e-11,
                                        "rtol": 1e-9, "max_step": b.MAX_STEP}).states[-1]
        self.assertLess((exact - reference).norm(), 2e-6)
        self.assertAlmostEqual(exact.tr(), 1, delta=1e-8)

    def test_direct_transmon_formula_matches_six_state_lindblad_average(self):
        g, e = qt.basis(2, 0), qt.basis(2, 1)
        collapse = [np.sqrt(1 / b.T1_T) * g * e.dag(),
                    np.sqrt(1 / (2 * b.TPHI_T)) * (e.proj() - g.proj())]
        times = np.array([0., 3., 25., 100.])
        values = []
        for target in (g, e, (g + e).unit(), (g - e).unit(),
                       (g + 1j * e).unit(), (g - 1j * e).unit()):
            result = qt.mesolve(0 * g.proj(), target.proj(), times, collapse,
                                e_ops=[target.proj()],
                                options={"atol": 1e-11, "rtol": 1e-9})
            values.append(result.expect[0])
        np.testing.assert_allclose(b.direct_transmon_fidelity(times),
                                   np.mean(values, axis=0), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
