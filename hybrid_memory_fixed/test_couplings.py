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
        config = validate({"PHI_SWAP": 0.2597137967167845, "BETA": .01,
                           "E_ZPF_V_PER_M": .08})
        result = derived(config)
        hbar = 6.62607015e-34 / (2 * math.pi)
        # Default flux preserves the 5000 MHz plateau.
        n01 = math.sqrt((5000+config["EC"])/(16*config["EC"]))
        self.assertAlmostEqual(result["transmon_charge_matrix_element"], n01)
        angular_tc = 2 * 1.602176634e-19 * 0.01 * 0.5e-6 * n01 / hbar
        angular_ac = 1000 * 1.602176634e-19 * 5.29177210544e-11 * 0.08 / hbar
        for prefix, angular in (("transmon", angular_tc), ("atom", angular_ac)):
            self.assertAlmostEqual(result[prefix + "_coupling_MHz"], angular / (2*math.pi*1e6))
            self.assertAlmostEqual(result[prefix + "_swap_hold_us"], math.pi / (2*angular) * 1e6)

    def test_each_physical_input_scales_the_coupling(self):
        base = coupling_strengths(DEFAULTS)
        for key, output in (("BETA", "transmon"), ("V_ZPF_UV", "transmon"),
                            ("D_01_EA0", "atom"),
                            ("E_ZPF_V_PER_M", "atom"), ("DIPOLE_ALIGNMENT", "atom")):
            with self.subTest(key=key):
                result = coupling_strengths(validate({key: DEFAULTS[key] / 2}))
                target = output + "_coupling_MHz"
                other = ("atom" if output == "transmon" else "transmon") + "_coupling_MHz"
                self.assertAlmostEqual(result[target], base[target] / 2)
                self.assertEqual(result[other], base[other])
        self.assertEqual(coupling_strengths(validate({"DIPOLE_ALIGNMENT": -1})), base)

    def test_charge_element_tracks_device_operating_point(self):
        base = derived({})
        for overrides in ({"FC": 4900}, {"PHI_SWAP": .3}, {"EC": 300},
                          {"EJ_MAX": 25000, "ASYMMETRY": .2}):
            with self.subTest(overrides=overrides):
                config = validate(overrides)
                result = derived(config)
                phi = math.pi*config["PHI_SWAP"]
                expected_ej = config["EJ_MAX"]*math.hypot(math.cos(phi), config["ASYMMETRY"]*math.sin(phi))
                expected_n = (expected_ej/(32*config["EC"]))**.25
                self.assertAlmostEqual(result["transmon_swap_EJ_MHz"], expected_ej)
                self.assertAlmostEqual(result["transmon_charge_matrix_element"], expected_n)
                self.assertAlmostEqual(result["transmon_coupling_MHz"] /
                                       base["transmon_coupling_MHz"],
                                       expected_n/base["transmon_charge_matrix_element"])
        self.assertNotIn("N_01", DEFAULTS)
        with self.assertRaisesRegex(ValueError, "N_01 is now calculated"):
            validate({"N_01": .8})

    def test_invalid_inputs_and_legacy_couplings_are_rejected(self):
        for values in ({"BETA": 1.1}, {"V_ZPF_UV": 0}, {"N_01": -1},
                       {"D_01_EA0": 0}, {"E_ZPF_V_PER_M": float("nan")},
                       {"DIPOLE_ALIGNMENT": 1.1}, {"DIPOLE_ALIGNMENT": 0},
                       {"G_TC": 20}, {"G_AC": 0.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate(values)

    def test_cavity_changes_do_not_retune_transmon(self):
        base = derived({})
        changed = derived({"FC": 4900})
        self.assertNotIn("F_T_INTERACTION", DEFAULTS)
        self.assertNotIn("F_SWAP_TARGET", DEFAULTS)
        for key in ("swap_flux", "swap_voltage_V", "transmon_coupling_MHz",
                    "atom_coupling_MHz", "transmon_swap_hold_us", "atom_swap_hold_us"):
            self.assertEqual(base[key], changed[key], key)
        for key in ("transmon_swap_detuning_MHz", "atom_swap_detuning_MHz"):
            self.assertAlmostEqual(changed[key] - base[key], 100, places=9)
        validate({"FC": 7000})
        with self.assertRaisesRegex(ValueError, "F_T_INTERACTION is now calculated"):
            validate({"F_T_INTERACTION": 5000})
        with self.assertRaisesRegex(ValueError, "F_SWAP_TARGET has been removed"):
            validate({"F_SWAP_TARGET": 5000})
        for flux in (-.01, .51, float("nan"), float("inf")):
            with self.subTest(flux=flux), self.assertRaises(ValueError):
                validate({"PHI_SWAP": flux})
        validate({"V_SWAP": -.3})

    def test_applied_flux_sets_resonance_and_device_parameters_change_it(self):
        for flux in (0, .2, .3, .5):
            for device in ({}, {"EC": 300}, {"EJ_MAX": 25000, "ASYMMETRY": .2}):
                with self.subTest(flux=flux, device=device):
                    config = validate({"PHI_SWAP": flux, "FC": 4950, **device})
                    ej = config["EJ_MAX"]*math.hypot(math.cos(math.pi*flux),
                                                   config["ASYMMETRY"]*math.sin(math.pi*flux))
                    frequency = math.sqrt(8*config["EC"]*ej)-config["EC"]
                    result = derived(config)
                    self.assertEqual(result["swap_flux"], flux)
                    self.assertAlmostEqual(result["transmon_interaction_frequency_MHz"], frequency)
                    self.assertAlmostEqual(result["transmon_swap_detuning_MHz"], frequency-4950)
        with self.assertRaisesRegex(ValueError, "nonpositive transmon frequency"):
            validate({"PHI_SWAP": .5, "ASYMMETRY": 0})

    def test_parked_flux_sets_idle_resonance(self):
        self.assertAlmostEqual(derived({})["transmon_park_frequency_MHz"], 5800)
        for flux in (0, .15, .3, .5):
            with self.subTest(flux=flux):
                result = derived({"PHI_PARK": flux})
                ej = DEFAULTS["EJ_MAX"]*math.hypot(math.cos(math.pi*flux),
                                                 DEFAULTS["ASYMMETRY"]*math.sin(math.pi*flux))
                self.assertAlmostEqual(result["transmon_park_frequency_MHz"],
                                       math.sqrt(8*DEFAULTS["EC"]*ej)-DEFAULTS["EC"])
                self.assertEqual(result["transmon_interaction_frequency_MHz"],
                                 derived({})["transmon_interaction_frequency_MHz"])
        for flux in (-.01, .51, float("nan"), float("inf")):
            with self.subTest(flux=flux), self.assertRaises(ValueError):
                validate({"PHI_PARK": flux})
        with self.assertRaisesRegex(ValueError, "F_T_PARK is now calculated"):
            validate({"F_T_PARK": 5800})
        with self.assertRaisesRegex(ValueError, "nonpositive transmon frequency"):
            validate({"PHI_PARK": .5, "ASYMMETRY": 0})

    def test_explicit_flux_and_voltage_changes_restore_resonance(self):
        config = validate({"FC": 4900, "PHI_SWAP": 0.27107257859935197})
        field = math.sqrt((config["FC"]-config["F_A_ZERO"])
                          /(config["K_R1"]-config["K_R0"]))
        config["V_SWAP"] = config["GAP_CM"]*(field-config["STRAY_FIELD"])
        result = derived(config)
        self.assertAlmostEqual(result["transmon_swap_detuning_MHz"], 0, places=9)
        self.assertAlmostEqual(result["atom_swap_detuning_MHz"], 0, places=9)

    def test_swap_frequencies_and_holds_follow_couplings(self):
        base = derived({})
        for parameter, prefix in (("BETA", "transmon"), ("D_01_EA0", "atom")):
            changed = derived({parameter: DEFAULTS[parameter]/2})
            frequency = prefix + "_swap_frequency_MHz"
            hold = prefix + "_swap_hold_us"
            self.assertAlmostEqual(changed[frequency], base[frequency]/2)
            self.assertAlmostEqual(changed[hold], 2*base[hold])
            self.assertAlmostEqual(changed[frequency]*changed[hold], .5)

    def test_applied_voltage_and_stray_field_set_atomic_frequency(self):
        for overrides in ({"V_SWAP": 0}, {"V_SWAP": -.2},
                          {"STRAY_FIELD": .7, "GAP_CM": .2},
                          {"F_A_ZERO": 4500}, {"K_R0": -2, "K_R1": -2}):
            with self.subTest(overrides=overrides):
                config = validate(overrides)
                result = derived(config)
                field = config["V_SWAP"]/config["GAP_CM"]+config["STRAY_FIELD"]
                expected = (config["F_A_ZERO"]
                            +(config["K_R1"]-config["K_R0"])*field**2-config["FC"])
                self.assertEqual(result["swap_voltage_V"], config["V_SWAP"])
                self.assertAlmostEqual(result["swap_field_V_per_cm"], field)
                self.assertAlmostEqual(result["atom_swap_detuning_MHz"], expected)

    def test_write_and_read_hamiltonians_preserve_control_detuning(self):
        for overrides in ({}, {"FC": 4900}, {"PHI_SWAP": .3}, {"PHI_PARK": .2},
                          {"FC": 4900, "PHI_SWAP": .3, "ASYMMETRY": .2},
                          {"STRAY_FIELD": .4, "GAP_CM": .2, "K_R1": -10}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "settings.json"
                path.write_text(json.dumps(overrides))
                spec = importlib.util.spec_from_file_location(
                    "detuning_test_backend", Path(__file__).with_name("backend.py"))
                b = importlib.util.module_from_spec(spec)
                with patch.object(sys, "argv", ["backend.py", "--config", str(path)]):
                    spec.loader.exec_module(b)
                d = derived(overrides)
                self.assertEqual(b.PHI_PARK, validate(overrides)["PHI_PARK"])
                self.assertAlmostEqual(b.transmon_frequency(b.PHI_PARK),
                                       d["transmon_park_frequency_MHz"])
                self.assertAlmostEqual(b.F_T_PARK, d["transmon_park_frequency_MHz"])
                self.assertAlmostEqual(b.flux(0), b.PHI_PARK)
                qt = b.qt
                excited = qt.tensor(b.e, qt.basis(b.NC, 0), b.r0)
                photon = qt.tensor(b.g, qt.basis(b.NC, 1), b.r0)
                atom = qt.tensor(b.g, qt.basis(b.NC, 0), b.r1)
                # Read and write must use exactly the same physical plateaus.
                read_h, end, stages, flux_read, voltage_read = b.retrieval_model()
                for hamiltonian, tc, ac, voltage in (
                        (b.H, b.TC, b.AC, b.electrode_voltage),
                        (read_h, stages[2][1], stages[1][1], voltage_read)):
                    t_tc, t_ac = sum(tc)/2, sum(ac)/2
                    self.assertAlmostEqual(voltage(t_ac), b.V_SWAP)
                    self.assertAlmostEqual(b.V_SWAP, d["swap_voltage_V"])
                    matrix = hamiltonian(t_tc)
                    dt = (qt.expect(matrix, excited)-qt.expect(matrix, photon))/b.TWOPI
                    self.assertAlmostEqual(dt, d["transmon_swap_detuning_MHz"], places=8)
                    self.assertAlmostEqual(b.transmon_frequency(b.PHI_SWAP),
                                           b.F_T_INTERACTION, places=8)
                    self.assertAlmostEqual(b.PHI_SWAP, d["swap_flux"])
                    self.assertEqual(b.PHI_SWAP, validate(overrides)["PHI_SWAP"])
                    self.assertAlmostEqual(b.F_T_INTERACTION, d["transmon_interaction_frequency_MHz"])
                    matrix = hamiltonian(t_ac)
                    da = (qt.expect(matrix, atom)-qt.expect(matrix, photon))/b.TWOPI
                    self.assertAlmostEqual(da, d["atom_swap_detuning_MHz"], places=8)
                    # Isolate the atom-photon exchange block at the plateau.
                    block = qt.Qobj([[0, b.G_AC], [b.G_AC, da*b.TWOPI]])
                    hold = math.pi/(2*b.G_AC)
                    final = (-1j*block*hold).expm()*qt.basis(2, 0)
                    transfer = abs(final.full()[1, 0])**2
                    if abs(da) < 1e-8:
                        self.assertAlmostEqual(transfer, 1)
                    else:
                        self.assertLessEqual(transfer,
                            4*b.G_AC**2/(4*b.G_AC**2+(da*b.TWOPI)**2)+1e-10)

    def test_detuned_cavity_suppresses_write_and_read_transfer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"detuned.json"
            # Stay far off resonance even with the larger calibrated couplings.
            path.write_text(json.dumps({"FC": 4500}))
            spec = importlib.util.spec_from_file_location(
                "detuned_transfer_backend", Path(__file__).with_name("backend.py"))
            b = importlib.util.module_from_spec(spec)
            with patch.object(sys, "argv", ["backend.py", "--config", str(path)]):
                spec.loader.exec_module(b)
            qt = b.qt
            _, written = b.simulate(0, 1, noisy=False, include_storage=False)
            self.assertLess(qt.expect(b.Ph1, written[-1]), .001)
            self.assertGreater(qt.expect(b.Pe, written[-1]), .99)
            self.assertLess(qt.expect(b.atomic_storage_effect(b.e, 0), written[-1]), .001)
            # Failed write/read can leave the input intact on the transmon.
            # That is a high return overlap, but must not count as atomic storage.
            _, bypass, _, _, _ = b.evolve_read(written[-1], noisy=False)
            returned_target = qt.tensor(b.e, qt.basis(b.NC, 0), b.h0)
            self.assertGreater(qt.expect(returned_target.proj(), bypass[-1]), .99)
            _, retrieved, _, _, _ = b.evolve_read(
                b.retrieval_initial(0, 1), noisy=False)
            target = qt.tensor(b.e, qt.basis(b.NC, 0), b.h0)
            self.assertLess(qt.expect(target.proj(), retrieved[-1]), .001)

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
            self.assertAlmostEqual(interval[1] - interval[0],
                                   d[prefix + "_pulse_duration_us"])
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
