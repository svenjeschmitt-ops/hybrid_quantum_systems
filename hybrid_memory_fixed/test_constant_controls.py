"""Optional constant controls must agree across write, read and reported settings."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hybrid_settings import DEFAULTS, derived, validate, effective_controls
from hybrid_panel import PARAMETER_SYMBOLS, parameter_symbol_png


class ConstantControlTests(unittest.TestCase):
    def test_options_default_off_and_allow_unreachable_resonance(self):
        self.assertEqual(DEFAULTS['CONSTANT_FLUX'], 0)
        self.assertEqual(DEFAULTS['CONSTANT_FIELD'], 0)
        for config in ({'CONSTANT_FIELD': 2}, {'CONSTANT_FLUX': -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate(config)
        # Constant controls must accept detuned and unreachable cavity frequencies.
        validate({'CONSTANT_FLUX': 1, 'FC': 10000})
        validate({'CONSTANT_FIELD': 1, 'F_A_ZERO': 4900})
        validate({'CONSTANT_FLUX': 1, 'F_A_ZERO': 4900})
        validate({'CONSTANT_FIELD': 1, 'FC': 8000, 'F_A_ZERO': 8150})

    def test_write_read_and_optical_carriers_use_each_selected_control(self):
        for constant_flux in (0, 1):
            for constant_field in (0, 1):
                with self.subTest(flux=constant_flux, field=constant_field), tempfile.TemporaryDirectory() as folder:
                    config = validate({'CONSTANT_FLUX': constant_flux, 'CONSTANT_FIELD': constant_field,
                                       'PHI_SWAP': .21, 'V_SWAP': .08,
                                       'STRAY_FIELD': .37, 'E_PARK': -.2, 'STORAGE': .02})
                    original = dict(config)
                    controls = effective_controls(config)
                    report = derived(config)
                    path = Path(folder)/'settings.json'
                    path.write_text(json.dumps(config))
                    spec = importlib.util.spec_from_file_location(
                        'constant_controls_backend', Path(__file__).with_name('backend.py'))
                    b = importlib.util.module_from_spec(spec)
                    with patch.object(sys, 'argv', ['backend.py', '--config', str(path)]):
                        spec.loader.exec_module(b)
                    _, _, stages, read_flux, read_voltage = b.retrieval_model()
                    for t in np.linspace(0, b.T_END, 101):
                        if constant_flux:
                            self.assertAlmostEqual(b.flux(t), config['PHI_SWAP'])
                            self.assertNotAlmostEqual(b.transmon_frequency(b.flux(t)), b.FC)
                            self.assertAlmostEqual(read_flux(t), b.flux(t))
                        if constant_field:
                            self.assertAlmostEqual(b.electrode_voltage(t), config['V_SWAP'])
                            self.assertNotAlmostEqual(b.F_A_ZERO+(b.K_R1-b.K_R0)*b.electric_field(t)**2, b.FC)
                            self.assertAlmostEqual(read_voltage(t), b.electrode_voltage(t))
                        self.assertAlmostEqual(b.electric_field(t), b.electrode_voltage(t)/b.GAP_CM+b.STRAY_FIELD)
                    if not constant_flux:
                        self.assertAlmostEqual(b.flux(0), config['PHI_PARK'])
                        self.assertAlmostEqual(b.flux(sum(b.TC)/2), config['PHI_SWAP'])
                        self.assertNotAlmostEqual(b.flux(0), b.flux(sum(b.TC)/2))
                    if not constant_field:
                        self.assertAlmostEqual(b.electric_field(0), config['E_PARK'])
                        self.assertAlmostEqual(b.electrode_voltage(sum(b.AC)/2), config['V_SWAP'])
                        self.assertNotAlmostEqual(b.electric_field(0), b.electric_field(sum(b.AC)/2))
                    self.assertAlmostEqual(b.G_TC/b.TWOPI, report['transmon_coupling_MHz'])
                    self.assertAlmostEqual(b.PHI_SWAP, report['swap_flux'])
                    self.assertAlmostEqual(b.E_SWAP, report['swap_field_V_per_cm'])
                    self.assertAlmostEqual(b.D_R0_PARK, b.TWOPI*b.K_R0*controls['E_PARK']**2)
                    self.assertAlmostEqual(b.D_R1_PARK, b.TWOPI*(b.F_A_ZERO-b.FC+b.K_R1*controls['E_PARK']**2))
                    self.assertEqual(config, original)

    def test_panel_symbols_render(self):
        for key in ('CONSTANT_FLUX', 'CONSTANT_FIELD'):
            self.assertIn(key, PARAMETER_SYMBOLS)
            self.assertTrue(parameter_symbol_png(key).startswith(b'\x89PNG'))


if __name__ == '__main__':
    unittest.main()
