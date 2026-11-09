"""Rounded control pulses and their use in the write/read Hamiltonians."""
import unittest
from unittest.mock import patch

import numpy as np
import qutip as qt

import backend as b


class RoundedPulseTests(unittest.TestCase):
    def test_controls_return_to_park_and_reach_the_interaction_value(self):
        _, _, stages, read_flux, read_voltage = b.retrieval_model()
        read_ac, read_tc = stages[1][1], stages[2][1]
        park_voltage = b.GAP_CM * (b.E_PARK - b.STRAY_FIELD)
        for control, interval, park, peak in (
            (b.flux, b.TC, b.PHI_PARK, b.PHI_SWAP),
            (b.electrode_voltage, b.AC, park_voltage, b.V_SWAP),
            (read_flux, read_tc, b.PHI_PARK, b.PHI_SWAP),
            (read_voltage, read_ac, park_voltage, b.V_SWAP),
        ):
            start, end = interval
            for t in (start - .01, start, end, end + .01):
                self.assertAlmostEqual(control(t), park)
            self.assertAlmostEqual(control((start + end) / 2), peak)
            # Quarter points distinguish a rounded excursion from a plateau.
            for fraction in (.25, .75):
                self.assertAlmostEqual(control(start + fraction * (end - start)),
                                       (park + peak) / 2)

    def test_endpoint_slopes_vanish_and_controls_do_not_overshoot(self):
        start, end = b.TC
        duration = end - start
        step = duration * 1e-7
        for edge, direction in ((start, 1), (end, -1)):
            slope = abs(b.smooth_window(edge + direction * step, start, end) / step)
            self.assertLess(slope * duration, 1e-5)
        values = np.array([b.smooth_window(t, start, end)
                           for t in np.linspace(start, end, 1001)])
        self.assertTrue(np.all((values >= 0) & (values <= 1)))
        np.testing.assert_allclose(values, values[::-1], atol=1e-14)

    def test_output_grids_include_control_peaks(self):
        for interval in (b.TC, b.AC):
            self.assertIn(sum(interval) / 2, b.time_grid(False))
        _, times, stages, _, _ = b.retrieval_model()
        for _, interval in stages[1:3]:
            self.assertIn(sum(interval) / 2, times)

    def test_electric_field_preserves_voltage_and_stray_field_relation(self):
        for t in np.linspace(0, b.MAP[1], 101):
            self.assertAlmostEqual(b.electric_field(t),
                                   b.electrode_voltage(t) / b.GAP_CM + b.STRAY_FIELD)
        self.assertAlmostEqual(b.electric_field(sum(b.AC) / 2), b.E_SWAP)

    def test_write_and_read_converge_when_solver_step_is_halved(self):
        initial = b.retrieval_initial(0, 1)
        _, written = b.simulate(0, 1, noisy=False, include_storage=False)
        _, read, _, _, _ = b.evolve_read(initial, noisy=False)
        with patch.object(b, "MAX_STEP", b.MAX_STEP / 2):
            _, refined_write = b.simulate(0, 1, noisy=False, include_storage=False)
            _, refined_read, _, _, _ = b.evolve_read(initial, noisy=False, step=b.MAX_STEP)
        for coarse, refined in ((written[-1], refined_write[-1]),
                                (read[-1], refined_read[-1])):
            self.assertAlmostEqual(coarse.norm(), 1, delta=1e-8)
            self.assertLess((coarse.proj() - refined.proj()).norm(), 2e-6)
        write_fidelity = qt.expect(b.atomic_storage_effect(b.e, 0), written[-1])
        target = qt.tensor(b.e, qt.basis(b.NC, 0), b.h0)
        print(f"Rounded pulses, noiseless excited-input joint fidelity: "
              f"write={write_fidelity:.6f}, read={qt.expect(target.proj(), read[-1]):.6f}")

    def test_calibrated_defaults_transfer_all_six_inputs_after_storage(self):
        # One phase reference per direction, shared by every input and noise mode.
        write_phase, read_phase = b.calibrated_phase(), b.read_phase()
        inputs = ((1, 0), (0, 1), (1, 1), (1, -1), (1, 1j), (1, -1j))
        for noisy in (False, True):
            minimum = .99 if noisy else .995
            for alpha, beta in inputs:
                with self.subTest(noisy=noisy, alpha=alpha, beta=beta):
                    target = (alpha*b.g + beta*b.e).unit()
                    _, states = b.simulate(alpha, beta, noisy=noisy, include_storage=False)
                    held = b.hold_memory(states[-1], noisy=noisy)
                    self.assertGreater(qt.expect(b.atomic_storage_effect(target, write_phase), held),
                                       minimum)
                    initial = b.stored_retrieval_initial(alpha, beta, noisy=noisy)
                    _, states, _, _, _ = b.evolve_read(initial, noisy=noisy)
                    self.assertGreater(b.read_metrics(states[-1:], target, read_phase)[2][-1],
                                       minimum)


if __name__ == "__main__":
    unittest.main()
