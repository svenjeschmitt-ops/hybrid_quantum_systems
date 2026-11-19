"""The timing search scores the actual complete readout for all six inputs."""
import unittest
from unittest.mock import patch

import numpy as np

import backend as b
from hybrid_settings import constant_field_intervals
from pulse_preview import control_preview


class ConstantFluxTimingTests(unittest.TestCase):
    def test_pulsed_controls_do_not_run_automatic_timing(self):
        class SimulationStarted(Exception):
            pass

        with patch.dict(b._SETTINGS, {"CONSTANT_FIELD": 0, "CONSTANT_FLUX": 0}), \
             patch.object(b, "_WRITE_TIMING_ACTIVE", False), \
             patch.object(b, "_READ_CLEANUP_START", .001), \
             patch.object(b, "automatic_constant_field_mapping") as field_timing, \
             patch.object(b, "estimate_constant_flux_write_timing") as flux_timing, \
             patch.object(b, "simulate", side_effect=SimulationStarted):
            with self.assertRaises(SimulationStarted):
                b.main()
            field_timing.assert_not_called()
            flux_timing.assert_not_called()
            _, _, stages, _, _ = b.retrieval_model()
            self.assertAlmostEqual(dict(stages)["Return atom to h0"][0],
                                   max(dict(stages)["Atom to cavity"][1],
                                       dict(stages)["Cavity to transmon"][1]) + b.GAP)

    def test_automatic_mapping_wait_uses_pilot_trajectory(self):
        with patch.dict(b._SETTINGS, {"CONSTANT_FIELD": 1, "CONSTANT_FLUX": 0}), \
             patch.object(b, "E_PARK", b.E_SWAP):
            result = b.automatic_constant_field_mapping()
            self.assertAlmostEqual(result["previous_pulse_end_us"], b.TC[1])
            self.assertGreaterEqual(result["wait_after_previous_pulse_us"], b.GAP)
            self.assertLessEqual(result["mapping_start_us"], result["search_end_us"])
            self.assertGreaterEqual(result["pilot_atomic_r1_population"], 0.)
            self.assertLessEqual(result["pilot_atomic_r1_population"], 1.)

    def test_constant_field_timing_and_preview_for_both_flux_modes(self):
        for constant_flux in (0, 1):
            with self.subTest(constant_flux=constant_flux), \
                 patch.dict(b._SETTINGS, {"CONSTANT_FIELD": 1, "CONSTANT_FLUX": constant_flux}), \
                 patch.object(b, "E_PARK", b.E_SWAP), \
                 patch.object(b, "PHI_PARK", b.PHI_SWAP if constant_flux else b.PHI_PARK), \
                 patch.object(b, "_READ_CLEANUP_START", None):
                try:
                    for direction in ("write", "read"):
                        preview = control_preview(b._SETTINGS, direction)
                        start = (b.PREP[1] + b.GAP if direction == "write" else
                                 max(b.T_OPT, b.T_OPT_1) + b.GAP)
                        tc, ac, mapping_start = constant_field_intervals(
                            b._SETTINGS, start, b.TC_DURATION, direction)
                        self.assertAlmostEqual(preview["stages"]["mapping"][0], mapping_start)
                        self.assertTrue(np.allclose(preview["voltage"], b.V_SWAP))
                        if direction == "write":
                            b.configure_write_atom_start(ac[0], mapping_start=mapping_start)
                            self.assertAlmostEqual(b.MAP[0], mapping_start)
                            self.assertLess(b.MAP[0], max(b.TC[1], b._NOMINAL_AC[1]) + b.GAP)
                        else:
                            _, times, stages, flux, voltage = b.retrieval_model()
                            self.assertAlmostEqual(dict(stages)["Return atom to h0"][0], mapping_start)
                            self.assertTrue(all(abs(voltage(t) - b.V_SWAP) < 1e-12 for t in times))
                            if constant_flux:
                                self.assertTrue(all(abs(flux(t) - b.PHI_SWAP) < 1e-12 for t in times))
                finally:
                    b.configure_write_atom_start(b._NOMINAL_AC[0], honor_tc=True)

    def test_coupling_estimate_starts_at_first_peak_after_preparation(self):
        with patch.object(b, "PHI_PARK", b.PHI_SWAP):
            timing = b.estimate_constant_flux_write_timing()
            start = timing["atom_start_us"]
            frequency = timing["exchange_frequency_MHz"]
            self.assertGreaterEqual(start, b.PREP[1] + b.GAP)
            self.assertLess(start - 1 / frequency, b.PREP[1] + b.GAP)
            self.assertAlmostEqual(np.sin(np.pi * frequency * start) ** 2, 1.)
            self.assertAlmostEqual(timing["write_time_us"],
                                   start + b.AC_DURATION + b.GAP + max(b.T_OPT, b.T_OPT_1))

    def test_write_search_matches_independent_complete_writes(self):
        inputs = ((1, 0), (0, 1), (1, 1), (1, -1), (1, 1j), (1, -1j))
        with patch.object(b, "PHI_PARK", b.PHI_SWAP):
            try:
                result = b.optimize_constant_flux_write(samples=2)
                b.configure_write_atom_start(result["atom_start_us"])
                self.assertAlmostEqual(b.MAP[1], result["write_time_us"])
                self.assertLessEqual(b.AC[0], b._NOMINAL_AC[0])
                self.assertTrue(all(abs(b.flux(t) - b.PHI_SWAP) < 1e-12
                                    for t in np.linspace(0, b.MAP[1], 20)))
                scores = []
                for alpha, beta in inputs:
                    _, states = b.simulate(alpha, beta, include_storage=False)
                    target = (alpha * b.g + beta * b.e).unit()
                    scores.append(float(np.real(b.qt.expect(b.atomic_storage_effect(target), states[-1]))))
                self.assertAlmostEqual(np.mean(scores), result["mean_six_state_joint_fidelity"], places=5)
            finally:
                b.configure_write_atom_start(b._NOMINAL_AC[0], honor_tc=True)

    def test_selected_score_matches_independent_complete_readouts(self):
        inputs = ((1, 0), (0, 1), (1, 1), (1, -1), (1, 1j), (1, -1j))
        with patch.dict(b._SETTINGS, {"CONSTANT_FLUX": 1, "CONSTANT_FIELD": 0}), \
             patch.object(b, "PHI_PARK", b.PHI_SWAP), patch.object(b, "_READ_CLEANUP_START", None):
            result = b.optimize_constant_flux_readout(samples=2)
            b._READ_CLEANUP_START = result["cleanup_start_us"]
            _, times, stages, flux, _ = b.retrieval_model()
            self.assertAlmostEqual(times[-1], result["retrieval_time_us"])
            self.assertTrue(all(abs(flux(t) - b.PHI_SWAP) < 1e-12 for t in times))
            self.assertGreaterEqual(result["cleanup_start_us"], dict(stages)["Atom to cavity"][1])
            scores = []
            for alpha, beta in inputs:
                initial = b.stored_retrieval_initial(alpha, beta)
                _, states, _, _, _ = b.evolve_read(initial)
                target = (alpha * b.g + beta * b.e).unit()
                scores.append(b.read_metrics(states[-1:], target)[1][0])
            self.assertAlmostEqual(np.mean(scores), result["mean_six_state_joint_fidelity"], places=5)


if __name__ == "__main__":
    unittest.main()
