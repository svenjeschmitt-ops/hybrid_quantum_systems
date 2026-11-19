"""Pulse-shape behavior across preview and simulation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from pulse_shapes import parse_shapes, pulse_window
from hybrid_settings import validate


class PulseShapesTests(unittest.TestCase):
    def test_default_matches_sin_squared_and_endpoints(self):
        t = np.linspace(-.2, 1.2, 101)
        actual = pulse_window(t, 0, 1, {}, 'flux')
        expected = np.where((t > 0) & (t < 1), np.sin(np.pi*t)**2, 0)
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_noise_is_reproducible_and_gaussians_change_area(self):
        spec = parse_shapes({'flux': {'base': 1, 'noise': .1, 'seed': 123,
            'gaussians': [[.4,.25,.08], [.7,.75,.05]]}})
        t = np.linspace(0, 1, 501)
        wave = pulse_window(t, 0, 1, spec, 'flux')
        np.testing.assert_array_equal(wave, pulse_window(t, 0, 1, spec, 'flux'))
        np.testing.assert_allclose(wave, [pulse_window(x, 0, 1, spec, 'flux') for x in t], atol=1e-14)
        np.testing.assert_allclose(wave[[0, -1]], [0, 0])
        self.assertGreater(abs(np.trapezoid(wave, t)-.5), .01)
        self.assertEqual(validate({'PULSE_SHAPES': spec})['PULSE_SHAPES'], json.dumps(spec, sort_keys=True))
        with self.assertRaises(ValueError):
            parse_shapes({'flux': {'gaussians': [[1, .5, 0]]}})

    def test_shaping_changes_actual_dynamics(self):
        script = r"""
import json
import numpy as np
import qutip as qt
import backend as b
_, states = b.simulate(1/np.sqrt(2), 1/np.sqrt(2), noisy=False, include_storage=False)
_, read, *_ = b.evolve_read(b.retrieval_initial(1, 1), noisy=False)
print(json.dumps([float(qt.expect(b.Ph1, states[-1])), float(qt.expect(b.Pe, read[-1]))]))
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            def run(config):
                path.write_text(json.dumps(config))
                proc = subprocess.run([sys.executable, '-c', script, '--config', str(path)],
                    cwd=Path(__file__).parent, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                return np.asarray(json.loads(proc.stdout.strip().splitlines()[-1]))
            baseline = run({'OMEGA': 10, 'OMEGA_1': 10})
            custom = {'OMEGA': 10, 'OMEGA_1': 10, 'PULSE_SHAPES': {'optical0': {'base': .5, 'noise': .05, 'seed': 7,
                                                       'gaussians': [[.5,.3,.1], [.3,.7,.08]]}}}
            shaped = run(custom)
        self.assertTrue(np.isfinite(shaped).all())
        self.assertGreater(np.max(np.abs(shaped-baseline)), 1e-4)

    def test_preview_matches_write_read_and_simultaneous_controls(self):
        script = r'''
import numpy as np
import backend as b
from pulse_preview import control_preview
from pulse_shapes import pulse_window
import simultaneous_coupling as sim
for direction in ('write', 'read'):
    view = control_preview(b._SETTINGS, direction)
    times = np.asarray(view['time_us'])
    if direction == 'write':
        flux, voltage = b.flux, b.electrode_voltage
    else:
        _, _, _, flux, voltage = b.retrieval_model()
    np.testing.assert_allclose(view['flux'], [flux(t) for t in times], atol=1e-12)
    np.testing.assert_allclose(view['voltage'], [voltage(t) for t in times], atol=1e-12)
    start = view['stages']['mapping'][0]
    for arm in (0,1):
        duration = b.T_OPT if arm == 0 else b.T_OPT_1
        interval = (start, start+duration) if (direction == 'write' or arm == 0) else (0., duration)
        midpoint = (interval[0]+interval[1])/2
        i = int(np.argmin(abs(times-midpoint)))
        expected = b.optical_envelope(times[i], *interval, arm=arm)/b.TWOPI
        # arm 0 may also have a non-overlapping prep/cleanup pulse.
        assert abs(view[f'optical{arm}'][i]-expected) < 1e-8
for direction in ('write', 'read'):
    _, _, info, flux, voltage, field = sim.model(b, include_storage=False, direction=direction)
    start, end = info['stages']['Simultaneous exchange']
    t = start + .37*(end-start)
    shape = pulse_window(t, start, end, b._PULSE_SHAPES, 'flux')
    expected = b.PHI_PARK+(info['interaction_flux']-b.PHI_PARK)*shape
    assert abs(flux(t)-expected) < 1e-12
assert b.MAX_STEP <= min(b.T_OPT,b.T_OPT_1)/128
'''
        shape = {'optical0': {'noise': .05, 'seed': 123, 'gaussians': [[.5,.3,.08]]},
                 'optical1': {'gaussians': [[.3,.7,.06]]},
                 'flux': {'noise': .08, 'seed': 9, 'gaussians': [[.2,.35,.1]]},
                 'voltage': {'gaussians': [[-.2,.6,.1]]}}
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/'config.json'
            config.write_text(json.dumps({'PULSE_SHAPES': shape}))
            proc = subprocess.run([sys.executable, '-c', script, '--config', str(config)],
                                  cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)


if __name__ == '__main__':
    unittest.main()
