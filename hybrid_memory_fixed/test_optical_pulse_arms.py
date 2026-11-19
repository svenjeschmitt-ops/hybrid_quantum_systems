"""Independent optical pulse controls, including the real protocol schedules."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hybrid_settings import derived, validate
from pulse_preview import control_preview
from optical_model import optical_detuning


class OpticalPulseArmTests(unittest.TestCase):
    def test_legacy_frequency_and_removed_detailed_model(self):
        self.assertEqual(validate({'OMEGA': 37})['OMEGA_1'], 37)
        self.assertEqual(validate({'OMEGA': 37, 'OMEGA_1': 82})['OMEGA_1'], 82)
        self.assertNotIn('OPTICAL_DETUNING_0', validate({'OPTICAL_DETUNING_0': 0}))
        migrated = validate({'OPTICAL_TRANSITION_0': 10000, 'OPTICAL_MAPPING_0': 10000, 'OPTICAL_DETUNING_0': 12})
        self.assertEqual(migrated['OPTICAL_MAPPING_0'], 10012)
        with self.assertRaisesRegex(ValueError, 'requires explicit'):
            validate({'OPTICAL_DETUNING_1': -12})
        with self.assertRaisesRegex(ValueError, 'OPTICAL_SCHEME has been removed'):
            validate({'OPTICAL_SCHEME': {'mode': 'effective'}})

    def test_measured_frequency_pairs_and_field_shift(self):
        self.assertIsNone(validate({})['OPTICAL_TRANSITION_0'])
        with self.assertRaisesRegex(ValueError, 'enter both'):
            validate({'OPTICAL_TRANSITION_0': 384000000})
        c = validate({'OPTICAL_TRANSITION_0': 384000000, 'OPTICAL_MAPPING_0': 384000000,
                      'OPTICAL_TRANSITION_1': 384005000, 'OPTICAL_MAPPING_1': 384005010})
        self.assertAlmostEqual(optical_detuning(c, 0, 0), 0)
        self.assertAlmostEqual(optical_detuning(c, 0, 2), -4*c['K_R0'])
        self.assertAlmostEqual(optical_detuning(c, 1, 2), 10-4*c['K_R1'])
        self.assertEqual(optical_detuning(validate({}), 0, 2), 0)

    def test_preview_and_durations(self):
        settings = {'OMEGA': 100, 'OMEGA_1': 50}
        values = derived(settings)
        self.assertAlmostEqual(values['optical_pi_pulse_us'], .01)
        self.assertAlmostEqual(values['optical_pi_pulse_1_us'], .02)
        write = control_preview(settings, 'write')['stages']
        read = control_preview(settings, 'read')['stages']
        self.assertAlmostEqual(write['preparation'][1], .01)
        self.assertAlmostEqual(write['mapping'][1] - write['mapping'][0], .02)
        self.assertAlmostEqual(read['preparation'][1], .02)
        self.assertAlmostEqual(read['mapping'][1] - read['mapping'][0], .01)

    def test_protocols_use_independent_pulse_shapes(self):
        script = r'''
import numpy as np
import qutip as qt
import backend as b
import simultaneous_coupling as sim
# With explicit laboratory-frame effective frequencies, changing the nominal
# field changes the atomic resonance, not the programmed mapping carrier.
b._SETTINGS['OPTICAL_TRANSITION_0'] = 384000000.
b._SETTINGS['OPTICAL_MAPPING_0'] = 384000020.
fixed0, _ = b.optical_protocol([(0, (0., b.T_OPT))], 0.)
fixed2, _ = b.optical_protocol([(0, (0., b.T_OPT))], 2.)
assert (fixed0(.003)-fixed2(.003)).norm() < 1e-9
b._SETTINGS['OPTICAL_TRANSITION_0'] = None
b._SETTINGS['OPTICAL_MAPPING_0'] = None
assert np.isclose(b.T_OPT, .01) and np.isclose(b.T_OPT_1, .02)
assert np.isclose(b.MAP[1]-b.MAP[0], .02)
assert np.isclose(b.MAP0[1]-b.MAP0[0], .01)
assert np.isclose(b.MAP1[1]-b.MAP1[0], .02)
h, times, stages, *_ = b.retrieval_model()
assert np.isclose(dict(stages)['Memory to Rydberg'][1], .02)
assert np.any(np.isclose(times, .01)) and np.any(np.isclose(times, .02))
for direction in ('write', 'read'):
    _, times, info, *_ = sim.model(b, include_storage=False, direction=direction)
    stage = info['stages']['Optical mapping' if direction == 'write' else 'Memory to Rydberg']
    assert np.isclose(stage[1]-stage[0], .02)
    assert np.any(np.isclose(times, stage[0]+.01))
# In an isolated resonant atom, each different-frequency pi pulse transfers
# its own ground state to its Rydberg state with high fidelity.
Hopt, _ = b.optical_protocol([(0, (0., b.T_OPT)), (1, (0., b.T_OPT_1))], b.E_PARK)
H = Hopt + b.TWOPI*b.atomic_energy_MHz('r0', b.E_PARK, offsets=False)*b.Pr0 + b.TWOPI*b.atomic_energy_MHz('r1', b.E_PARK, offsets=False)*b.Pr1
for arm, initial, target in [(0,b.h0,b.r0),(1,b.h1,b.r1)]:
    psi = qt.tensor(b.g, qt.basis(b.NC,0), initial)
    goal = qt.tensor(b.g, qt.basis(b.NC,0), target)
    result = qt.sesolve(H, psi, [0., .02], options={'max_step': .00002, 'atol': 1e-10, 'rtol': 1e-9})
    assert abs(goal.overlap(result.states[-1]))**2 > .9999
# Shift each carrier independently: a large mismatch suppresses mapping in
# both directions while the other arm remains resonant.
for arm in (0, 1):
    for index in (0, 1):
        b._SETTINGS[f'OPTICAL_TRANSITION_{index}'] = 384000000.
        b._SETTINGS[f'OPTICAL_MAPPING_{index}'] = 384001000. if arm == index else 384000000.
    optical, _ = b.optical_protocol([(0, (0., b.T_OPT)), (1, (0., b.T_OPT_1))], b.E_PARK)
    H = optical + b.TWOPI*b.atomic_energy_MHz('r0', b.E_PARK, offsets=False)*b.Pr0 + b.TWOPI*b.atomic_energy_MHz('r1', b.E_PARK, offsets=False)*b.Pr1
    for index, lower, upper in [(0,b.h0,b.r0),(1,b.h1,b.r1)]:
        for start, finish in ((lower,upper), (upper,lower)):
            psi = qt.tensor(b.g, qt.basis(b.NC,0), start)
            goal = qt.tensor(b.g, qt.basis(b.NC,0), finish)
            result = qt.sesolve(H, psi, [0., .02], options={'max_step': .00002, 'atol': 1e-10, 'rtol': 1e-9})
            population = abs(goal.overlap(result.states[-1]))**2
            assert population < .01 if arm == index else population > .9999, (arm, index, population)
'''
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'settings.json'
            config.write_text(json.dumps({'OMEGA': 100, 'OMEGA_1': 50, 'REL_WRONG_DIPOLE': 0, 'RYDBERG_MANIFOLD': []}))
            proc = subprocess.run([sys.executable, '-c', script, '--config', str(config)],
                                  cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == '__main__':
    unittest.main()
