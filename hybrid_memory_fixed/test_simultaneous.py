"""Physical and integration checks for simultaneous shared-cavity exchange."""
import unittest
from unittest.mock import patch

import numpy as np
import qutip as qt

import backend as b
from hybrid_settings import validate, calculated_transmon_frequency
from hybrid_panel import MODES, SECTION_EQUATIONS, parameter_symbol_png, section_equations_png
from simultaneous_coupling import model, plateau, resonance_settings, simulate


class SimultaneousTests(unittest.TestCase):
    def test_resonance_and_unreachable_inputs(self):
        c = validate({})
        info = resonance_settings(c)
        self.assertAlmostEqual(calculated_transmon_frequency(c, info['resonant_flux']), c['FC'])
        self.assertAlmostEqual(c['F_A_ZERO']+(c['K_R1']-c['K_R0'])*info['resonant_field_V_per_cm']**2, c['FC'])
        self.assertEqual(c, validate({}))
        for changes in ({'FC': 10000}, {'F_A_ZERO': 4900}, {'K_R1': -2}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'Simultaneous resonance'):
                resonance_settings(changes)
        info = resonance_settings({'K_R1': -2, 'F_A_ZERO': 5000})
        self.assertEqual(info['resonant_field_V_per_cm'], c['E_PARK'])

    def test_shared_plateau_and_return_to_park(self):
        h, ts, info, flux, _, field = model(b)
        start, end = info['stages']['Simultaneous exchange']
        for t in (start+info['ramp_us'], (start+end)/2, end-info['ramp_us']):
            self.assertAlmostEqual(b.transmon_frequency(flux(t)), b.FC)
            self.assertAlmostEqual(b.F_A_ZERO+(b.K_R1-b.K_R0)*field(t)**2, b.FC)
            self.assertTrue(h(t).isherm)
        for t in (0, start, end, ts[-1]):
            self.assertEqual(flux(t), b.PHI_PARK)
            self.assertEqual(field(t), b.E_PARK)
        self.assertIn(info['mapping_end_us'], ts)
        self.assertEqual(plateau(-1, 0, 1, .1), 0)

    def test_flat_hold_reproduces_analytic_transfer_limit(self):
        # Independent analytic result in the single-excitation manifold.
        # Cover matched and strongly mismatched cavity couplings.
        reference = resonance_settings(b._SETTINGS)
        matched = b._SETTINGS['BETA']*reference['g_AC_rad_per_us']/reference['g_TC_rad_per_us']
        for beta in (.005, matched):
            with patch.dict(b._SETTINGS, BETA=beta):
                h, _, info, *_ = model(b)
            start, end = info['stages']['Simultaneous exchange']
            initial = qt.tensor(b.e, qt.basis(b.NC, 0), b.r0)
            target = qt.tensor(b.g, qt.basis(b.NC, 0), b.r1)
            photon = qt.tensor(b.g, qt.basis(b.NC, 1), b.r0)
            matrix = h((start+end)/2)
            final = (-1j*matrix*info['reference_hold_us']).expm()*initial
            self.assertAlmostEqual(abs(target.overlap(final))**2, info['ideal_transfer_limit'], places=10)
            self.assertLess(abs(photon.overlap(final))**2, 1e-15)
            self.assertAlmostEqual(matrix.matrix_element(photon.dag(), initial), info['g_TC_rad_per_us'])
            self.assertAlmostEqual(matrix.matrix_element(target.dag(), initial), 0)

    def test_full_protocol_preserves_complex_input_and_physical_state(self):
        with patch.object(b, 'STORAGE', .02):
            _, calibration, _ = simulate(b, 1, 1, noisy=False, include_storage=False)
            phase = np.angle(calibration[-1].ptrace(2)[0, 1])
            ts, states, info = simulate(b, 1/np.sqrt(2), 1j/np.sqrt(2), noisy=True)
        target = qt.tensor(b.g, qt.basis(b.NC, 0),
                           b.memory_rotation(phase).dag()*(b.h0+1j*b.h1).unit())
        # Compare the constant-generator storage with direct Lindblad evolution.
        h, _, _, *_ = model(b, include_storage=False)
        mapping_index = np.argmin(abs(ts-info['mapping_end_us']))
        direct = qt.mesolve(h(info['mapping_end_us']), states[mapping_index], [0, .02],
                            c_ops=b.C_OPS, options={'atol': 1e-11, 'rtol': 1e-9,
                                                   'method': 'dop853', 'nsteps': 100000})
        self.assertLess((direct.states[-1]-states[-1]).norm(), 1e-7)
        fidelity = qt.expect(target.proj(), states[-1])
        self.assertGreater(fidelity, .90)
        self.assertAlmostEqual(states[-1].tr(), 1, places=8)
        self.assertLess((states[-1]-states[-1].dag()).norm(), 1e-8)
        self.assertGreater(states[-1].eigenenergies().min(), -1e-8)
        self.assertAlmostEqual(ts[-1], info['mapping_end_us']+.02)

    def test_panel_mode_and_equations_render(self):
        self.assertEqual(dict(MODES)['Simultaneous cavity resonance: transmon to atom'], ['--simultaneous'])
        self.assertIn('Simultaneous resonance', SECTION_EQUATIONS)
        for key in ('SIM_RAMP', 'SIM_HOLD_SCALE'):
            self.assertTrue(parameter_symbol_png(key).startswith(b'\x89PNG'))
        self.assertTrue(section_equations_png('Simultaneous resonance').startswith(b'\x89PNG'))


if __name__ == '__main__':
    unittest.main()
