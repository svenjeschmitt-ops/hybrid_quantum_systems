"""The timing search scores the actual complete readout for all six inputs."""
import unittest
from unittest.mock import patch

import numpy as np

import backend as b
from hybrid_settings import constant_field_intervals
from pulse_preview import control_preview


class ConstantFluxTimingTests(unittest.TestCase):
    def test_atom_pulse_starts_at_coupling_switch_and_mapping_is_searched(self):
        inputs = ((1,0), (0,1), (1,1), (1,1j))
        with patch.dict(b._SETTINGS, {"CONSTANT_FLUX": 1, "CONSTANT_FIELD": 0}), \
             patch.object(b, "PHI_PARK", b.PHI_SWAP), patch.object(b, "_WRITE_GATE_TIME", None):
            try:
                b._WRITE_GATE_TIME = b.cavity_gate_time('write')['time_us']
                result = b.optimize_constant_field_write(samples=2, atom_start=b._WRITE_GATE_TIME)
                self.assertAlmostEqual(result['atom_start_us'], b._WRITE_GATE_TIME)
                self.assertGreaterEqual(result['mapping_start_us'], result['atom_start_us']+b.AC_DURATION+b.GAP-1e-12)
                b.configure_write_atom_start(result['atom_start_us'], mapping_start=result['mapping_start_us'])
                finals = [b.simulate(a,c,include_storage=False)[1][-1] for a,c in inputs]
                self.assertAlmostEqual(b._six_state_mean(finals,'write'), result['mean_six_state_joint_fidelity'], places=5)
                self.assertTrue(all(abs(b.flux(t)-b.PHI_SWAP)<1e-12 for t in np.linspace(0,b.MAP[1],30)))
            finally:
                b._WRITE_GATE_TIME = None
                b.configure_write_atom_start(b._NOMINAL_AC[0],honor_tc=True)

    def test_switchable_destination_couplings_are_off_then_on(self):
        tc = b.G_TC*(b.a.dag()*b.sm+b.a*b.sm.dag())
        ac = b.G_AC*(b.a*b.Rplus+b.a.dag()*b.Rplus.dag())
        for direction, expected in (("write", tc), ("read", ac)):
            evolution = b.qt.QobjEvo(b.transfer_couplings(direction, .05))
            self.assertLess((evolution(.049)-expected).norm(), 1e-10)
            self.assertLess((evolution(.051)-tc-ac).norm(), 1e-10)
        self.assertLess((b.qt.QobjEvo(b.transfer_couplings('write', None))(0)-tc-ac).norm(), 1e-10)

    def test_gate_is_included_in_solver_time_points(self):
        self.assertEqual(b.integration_points(0., 1., .4), [0., .4, 1.])
        self.assertEqual(b.integration_points(.5, 1., .4), [.5, 1.])

    def test_cavity_peak_uses_sampled_photons_before_mapping(self):
        times = np.array([0., .1, .2, .3, .4])
        photons = np.array([0., .2, .4, .1, .8])
        peak = b.sampled_cavity_peak(times, photons, .1, .3)
        self.assertEqual(peak['time_us'], .2)
        self.assertEqual(peak['mean_photons'], .4)
        self.assertIsNone(b.sampled_cavity_peak(times, np.zeros(5), .1, .3))

    def test_sampled_constant_field_write_and_read_scores(self):
        inputs = ((1, 0), (0, 1), (1, 1), (1, 1j))
        for fixed_flux in (0, 1):
            with self.subTest(fixed_flux=fixed_flux), \
                 patch.dict(b._SETTINGS, {"CONSTANT_FIELD": 1, "CONSTANT_FLUX": fixed_flux}), \
                 patch.object(b, "E_PARK", b.E_SWAP), \
                 patch.object(b, "PHI_PARK", b.PHI_SWAP if fixed_flux else b.PHI_PARK), \
                 patch.object(b, "_READ_CLEANUP_START", None), \
                 patch.object(b, "_READ_TC_START", None), \
                 patch.object(b, "_WRITE_GATE_TIME", None), \
                 patch.object(b, "_READ_GATE_TIME", None):
                try:
                    b._WRITE_GATE_TIME = b.cavity_gate_time('write')['time_us']
                    b.configure_write_atom_start(b._NOMINAL_AC[0],honor_tc=True)
                    write = b.optimize_constant_field_write(samples=2)
                    b.configure_write_atom_start(write['atom_start_us'], mapping_start=write['mapping_start_us'])
                    self.assertTrue(all(abs(b.electrode_voltage(t)-b.V_SWAP)<1e-12
                                        for t in np.linspace(0,b.MAP[1]+b.STORAGE,30)))
                    if fixed_flux:
                        self.assertTrue(all(abs(b.flux(t)-b.PHI_SWAP)<1e-12
                                            for t in np.linspace(0,b.MAP[1]+b.STORAGE,30)))
                    finals = [b.simulate(a,c,include_storage=False)[1][-1] for a,c in inputs]
                    self.assertAlmostEqual(b._six_state_mean(finals,'write'),write['mean_six_state_joint_fidelity'],places=5)
                    b._READ_GATE_TIME = b.cavity_gate_time('read')['time_us']
                    read = b.optimize_constant_field_readout(samples=2)
                    b._READ_CLEANUP_START = read['cleanup_start_us']
                    b._READ_TC_START = read['tc_start_us']
                    finals = [b.evolve_read(b.stored_retrieval_initial(a,c))[1][-1] for a,c in inputs]
                    self.assertAlmostEqual(b._six_state_mean(finals,'read'),read['mean_six_state_joint_fidelity'],places=5)
                    times,_,_,flux,voltage = b.evolve_read(b.stored_retrieval_initial(1,0))
                    self.assertAlmostEqual(times[-1],read['retrieval_time_us'])
                    self.assertTrue(all(abs(voltage(t)-b.V_SWAP)<1e-12 for t in times))
                    if fixed_flux:
                        self.assertTrue(all(abs(flux(t)-b.PHI_SWAP)<1e-12 for t in times))
                finally:
                    b._WRITE_GATE_TIME = None
                    b._READ_GATE_TIME = None
                    b.configure_write_atom_start(b._NOMINAL_AC[0],honor_tc=True)

    def test_pulsed_controls_do_not_run_automatic_timing(self):
        class SimulationStarted(Exception):
            pass

        with patch.dict(b._SETTINGS, {"CONSTANT_FIELD": 0, "CONSTANT_FLUX": 0}), \
             patch.object(b, "_WRITE_TIMING_ACTIVE", False), \
             patch.object(b, "_READ_CLEANUP_START", .001), \
             patch.object(b, "automatic_constant_field_mapping") as field_timing, \
             patch.object(b, "estimate_constant_flux_write_timing") as flux_timing, \
             patch.object(b, "optimize_constant_field_write") as field_search, \
             patch.object(b, "optimize_constant_flux_write") as flux_search, \
             patch.object(b, "simulate", side_effect=SimulationStarted):
            with self.assertRaises(SimulationStarted):
                b.main()
            field_timing.assert_not_called()
            flux_timing.assert_not_called()
            field_search.assert_not_called()
            flux_search.assert_not_called()
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
        with patch.dict(b._SETTINGS, {"CONSTANT_FLUX": 1, "CONSTANT_FIELD": 0}), \
             patch.object(b, "PHI_PARK", b.PHI_SWAP), patch.object(b, "_WRITE_GATE_TIME", None):
            try:
                b._WRITE_GATE_TIME = b.cavity_gate_time('write')['time_us']
                b.configure_write_atom_start(b._NOMINAL_AC[0],honor_tc=True)
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
                b._WRITE_GATE_TIME = None
                b.configure_write_atom_start(b._NOMINAL_AC[0], honor_tc=True)

    def test_selected_score_matches_independent_complete_readouts(self):
        inputs = ((1, 0), (0, 1), (1, 1), (1, -1), (1, 1j), (1, -1j))
        with patch.dict(b._SETTINGS, {"CONSTANT_FLUX": 1, "CONSTANT_FIELD": 0}), \
             patch.object(b, "PHI_PARK", b.PHI_SWAP), patch.object(b, "_READ_CLEANUP_START", None), \
             patch.object(b, "_READ_GATE_TIME", None):
            b._READ_GATE_TIME = b.cavity_gate_time('read')['time_us']
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
