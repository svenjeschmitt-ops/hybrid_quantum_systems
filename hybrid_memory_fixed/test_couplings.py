"""Coupling formulas, settings validation, and Hamiltonian unit checks.

Run: python -m unittest discover -s hybrid_memory_fixed -p 'test_*.py'
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hybrid_settings import DEFAULTS, coupling_strengths, derived, load, validate


class CouplingTests(unittest.TestCase):
    def test_si_formulas_and_swap_times(self):
        config = validate({})
        result = derived(config)
        hbar = 6.62607015e-34 / (2 * math.pi)
        angular_tc = 2 * 1.602176634e-19 * 0.1 * 0.5e-6 * 0.8 / hbar
        angular_ac = 1000 * 1.602176634e-19 * 5.29177210544e-11 * 0.04 / hbar
        for prefix, angular in (("transmon", angular_tc), ("atom", angular_ac)):
            self.assertAlmostEqual(result[prefix + "_coupling_MHz"], angular / (2*math.pi*1e6))
            self.assertAlmostEqual(result[prefix + "_swap_hold_us"], math.pi / (2*angular) * 1e6)

    def test_each_physical_input_scales_the_coupling(self):
        base = coupling_strengths(DEFAULTS)
        for key, output in (("BETA", "transmon"), ("V_ZPF_UV", "transmon"),
                            ("N_01", "transmon"), ("D_01_EA0", "atom"),
                            ("E_ZPF_V_PER_M", "atom"), ("DIPOLE_ALIGNMENT", "atom")):
            with self.subTest(key=key):
                result = coupling_strengths(validate({key: DEFAULTS[key] / 2}))
                target = output + "_coupling_MHz"
                other = ("atom" if output == "transmon" else "transmon") + "_coupling_MHz"
                self.assertAlmostEqual(result[target], base[target] / 2)
                self.assertEqual(result[other], base[other])
        self.assertEqual(coupling_strengths(validate({"DIPOLE_ALIGNMENT": -1})), base)

    def test_invalid_inputs_and_legacy_couplings_are_rejected(self):
        for values in ({"BETA": 1.1}, {"V_ZPF_UV": 0}, {"N_01": -1},
                       {"D_01_EA0": 0}, {"E_ZPF_V_PER_M": float("nan")},
                       {"DIPOLE_ALIGNMENT": 1.1}, {"DIPOLE_ALIGNMENT": 0},
                       {"G_TC": 20}, {"G_AC": 0.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate(values)

    def test_saved_settings_drive_hamiltonian_and_stage_timing(self):
        config = validate({"BETA": 0.15, "DIPOLE_ALIGNMENT": -0.5})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps(config))
            self.assertEqual(load(path), config)
            spec = importlib.util.spec_from_file_location(
                "coupling_test_backend", Path(__file__).with_name("backend.py"))
            backend = importlib.util.module_from_spec(spec)
            with patch.object(sys, "argv", ["backend.py", "--config", str(path)]):
                spec.loader.exec_module(backend)
        d = derived(config)
        for interval, prefix in ((backend.TC, "transmon"), (backend.AC, "atom")):
            self.assertAlmostEqual(interval[1] - interval[0] - 2*backend.RAMP,
                                   d[prefix + "_swap_hold_us"])
        qt = backend.qt
        initial = qt.tensor(backend.e, qt.basis(backend.NC, 0), backend.r0)
        photon = qt.tensor(backend.g, qt.basis(backend.NC, 1), backend.r0)
        atomic = qt.tensor(backend.g, qt.basis(backend.NC, 0), backend.r1)
        self.assertAlmostEqual(backend.H_coupling.matrix_element(photon.dag(), initial),
                               2*math.pi*d["transmon_coupling_MHz"])
        self.assertAlmostEqual(backend.H_coupling.matrix_element(atomic.dag(), photon),
                               2*math.pi*d["atom_coupling_MHz"])


if __name__ == "__main__":
    unittest.main()
