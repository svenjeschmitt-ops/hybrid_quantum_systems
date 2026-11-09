"""Static experimental errors must change both Hamiltonians, not calibration."""
import unittest
from unittest.mock import patch

import tolerance_bands

import numpy as np

import backend as b
from hybrid_settings import validate
from tolerance_scan import AXES, fixed_offsets, scenarios, summarize


class ToleranceTests(unittest.TestCase):
    def test_grid_is_complete_unique_and_nominal_first(self):
        points = list(scenarios({"FC_TOL": 2, "FLUX_TOL": .01}))
        self.assertEqual(len(points), 9)
        self.assertFalse(any(points[0].values()))
        self.assertEqual({(p['FC'], p['FLUX']) for p in points},
                         {(f, phi) for f in (-2, 0, 2) for phi in (-.01, 0, .01)})
        self.assertEqual(len(list(scenarios({}))), 1)

    def test_invalid_tolerances_rejected(self):
        for _, key, _ in AXES:
            for value in (-.1, float('inf'), float('nan')):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate({key: value})

    def test_write_read_and_storage_hamiltonians_use_same_static_errors(self):
        offsets = dict(FC=1.2, FT=-.7, FA=.3, FLUX=.002, EFIELD=.04)
        t = b.MAP[1]  # parked; also the Hamiltonian used by storage
        nominal = b.H(t)
        nominal_read = b.retrieval_model()[0](0)
        expected = (
            b.TWOPI * offsets['FC'] * b.n
            + b.TWOPI * (b.transmon_frequency(b.PHI_PARK + offsets['FLUX'])
                        - b.transmon_frequency(b.PHI_PARK) + offsets['FT']) * b.Pe
            + b.TWOPI * b.K_R0 * ((b.E_PARK + offsets['EFIELD'])**2 - b.E_PARK**2) * b.Pr0
            + b.TWOPI * (offsets['FA'] + b.K_R1 * ((b.E_PARK + offsets['EFIELD'])**2 - b.E_PARK**2)) * b.Pr1
            + b.TWOPI * b.K_WRONG * ((b.E_PARK + offsets['EFIELD'])**2 - b.E_PARK**2) * b.Pwrong
        )
        with fixed_offsets(b, offsets):
            self.assertLess((b.H(t) - nominal - expected).norm(), 1e-8)
            self.assertLess((b.retrieval_model()[0](0) - nominal_read - expected).norm(), 1e-8)
            for time in (0, sum(b.AC)/2, t):
                commanded = b.electrode_voltage(time)/b.GAP_CM + b.STRAY_FIELD
                self.assertAlmostEqual(b.electric_field(time), commanded + offsets['EFIELD'])
        self.assertLess((b.H(t) - nominal).norm(), 1e-10)

    def test_offsets_and_parked_solver_cache_restore_on_failure(self):
        before = dict(b.TOLERANCE_OFFSETS)
        b.parked_storage_solver()
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with fixed_offsets(b, {'FC': 2}):
                self.assertEqual(b.parked_storage_solver.cache_info().currsize, 0)
                raise RuntimeError('test')
        self.assertEqual(b.TOLERANCE_OFFSETS, before)
        self.assertEqual(b.parked_storage_solver.cache_info().currsize, 0)

    def test_detuning_bands_reference_the_actual_cavity(self):
        times = np.array([0., .1, .2])
        for mode in ('write', 'read'):
            nominal = tolerance_bands.controls(b, times, mode, 'hyperfine')
            for dc in (-100., 100.):
                with fixed_offsets(b, {'FC': dc}):
                    shifted = tolerance_bands.controls(b, times, mode, 'hyperfine')
                for key in ('transmon_detuning', 'atom_detuning'):
                    np.testing.assert_allclose(shifted[key], nominal[key] - dc)
                np.testing.assert_array_equal(shifted['flux'], nominal['flux'])
                np.testing.assert_array_equal(shifted['field'], nominal['field'])

    def test_zero_tolerances_do_not_run_extra_trajectories(self):
        with patch.object(b, 'simulate') as simulate, patch.object(b, 'evolve_read') as read:
            result = tolerance_bands.compute(b, validate({}), 'write', 1, 0, 0,
                                            None, None, None, None)
        self.assertIsNone(result)
        simulate.assert_not_called()
        read.assert_not_called()

    def test_summary_metrics_use_the_correct_subsystem_and_combined_population(self):
        write = b.qt.tensor(b.g, b.qt.basis(b.NC, 0), (b.h0 + 1j*b.h1).unit())
        read = b.qt.tensor((b.g + b.e).unit(), b.qt.basis(b.NC, 0), b.h0)
        for mode, state in (('write', write), ('read', read)):
            values = tolerance_bands.final_metrics(b, state, mode, .7, .8, [.9, .85, .95])
            self.assertAlmostEqual(values['coherence'], 1)
            self.assertEqual(values['basis_1'], .85)
            self.assertEqual(values['final_fidelity'], .7)
        mix = .3*b.qt.tensor(b.g, b.qt.basis(b.NC, 0), b.r0).proj() + \
              .7*b.qt.tensor(b.g, b.qt.basis(b.NC, 0), b.r1).proj()
        values = tolerance_bands.final_metrics(b, mix, 'write', 0, 0, [0, 0, 0])
        self.assertAlmostEqual(values['rydberg_population'], 1)

    def test_each_summary_row_has_a_separate_range_column(self):
        timing = [['Time', '1 ns'] for _ in range(5)]
        rows = timing + [[str(i), '0.5'] for i in range(11)]
        keys = ('basis_0', 'basis_1', 'basis_plus', 'before_fidelity', 'final_fidelity',
                'coherence', 'transmon_e', 'cavity_mean_photons', 'rydberg_population',
                'atom_loss', 'atom_wrong_rydberg')
        bounds = {key: {'low': .1, 'high': .9} for key in keys}
        bounds['before_fidelity'] = {'low': .2, 'high': .8}
        bands = {'summary': {'metric_ranges': bounds}}
        result = tolerance_bands.table_rows(rows, bands, 'write')
        self.assertEqual(len(result), len(rows))
        self.assertTrue(all(len(row) == 3 for row in result))
        self.assertTrue(all(row[2] == '—' for row in result[:5]))
        self.assertEqual(result[5][2], '10.00%–90.00%')
        self.assertEqual(result[8][2], '20.00%–80.00%')
        self.assertEqual(result[9][2], '10.00%–90.00%')
        self.assertEqual(result[13][2], '1.000e-01–9.000e-01')
        zero = tolerance_bands.table_rows(rows, None, 'write')
        self.assertEqual(zero[5][2], '0.5–0.5')

    def test_summary_selects_each_direction_independently(self):
        rows = [dict(write_fidelity=.8, read_fidelity=.7, offsets={}),
                dict(write_fidelity=.9, read_fidelity=.6, offsets={'FC': 1}),
                dict(write_fidelity=.6, read_fidelity=.9, offsets={'FC': -1})]
        result = summarize(rows)
        self.assertIs(result['write']['best'], rows[1])
        self.assertIs(result['write']['worst'], rows[2])
        self.assertIs(result['read']['best'], rows[2])
        self.assertIs(result['read']['worst'], rows[1])


if __name__ == '__main__':
    unittest.main()
