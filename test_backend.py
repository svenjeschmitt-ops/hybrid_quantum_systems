"""Physical matrix-element and solver checks for the Hamiltonian builder."""

import unittest

import numpy as np
import qutip as qt

from backend import (
    ATOM_A, ATOM_B, ATOM_R, ATOM_R_TIL,
    build_hamiltonian, make_rydberg_laser, quantumsystem,
)


def state(photons, transmon, atom, cutoffs=(2,), nt=2):
    return qt.tensor(
        *[qt.basis(d, n) for d, n in zip(cutoffs, photons)],
        qt.basis(nt, transmon), qt.basis(4, atom),
    )


class HamiltonianTests(unittest.TestCase):
    def test_free_energies_and_dimensions(self):
        H = build_hamiltonian([2, 3], [5, 7], [0, 11], [0, 2, 17, 19],
                              np.zeros((2, 1)))
        self.assertEqual(H.dims, [[2, 3, 2, 4], [2, 3, 2, 4]])
        psi = state([1, 2], 1, ATOM_R_TIL, cutoffs=(2, 3))
        self.assertAlmostEqual(qt.expect(H(0), psi), 5 + 14 + 11 + 19)

    def test_complex_exchange_and_independent_modes(self):
        g_t, g_a = 0.3 + 0.2j, 0.4 - 0.1j
        H = build_hamiltonian(
            [2, 2], [5, 7], [0, 11], [0, 2, 17, 19], [[g_t], [0]],
            atom_couplings=[(1, ATOM_R, ATOM_R_TIL, g_a)],
        )(0)
        initial_t = state([0, 0], 1, ATOM_A, (2, 2))
        final_t = state([1, 0], 0, ATOM_A, (2, 2))
        wrong_mode = state([0, 1], 0, ATOM_A, (2, 2))
        self.assertAlmostEqual(H.matrix_element(final_t.dag(), initial_t), g_t)
        self.assertAlmostEqual(H.matrix_element(initial_t.dag(), final_t), g_t.conjugate())
        self.assertAlmostEqual(H.matrix_element(wrong_mode.dag(), initial_t), 0)
        initial_a = state([0, 0], 0, ATOM_R_TIL, (2, 2))
        final_a = state([0, 1], 0, ATOM_R, (2, 2))
        self.assertAlmostEqual(H.matrix_element(final_a.dag(), initial_a), g_a)
        self.assertTrue(H.isherm)

    def test_drives_have_correct_phases_and_conjugates(self):
        lasers = make_rydberg_laser(3, lambda t: 2 + 1j, lambda t: 0.4j)
        cavity_f = lambda t: (0.7 - 0.2j) * np.exp(-2j * t)
        H = build_hamiltonian(
            [2], [5], [0, 11], [0, 2, 17, 19], [[0]],
            cavity_drives=[(0, cavity_f)], laser_drives=lasers,
        )
        ground = state([0], 0, ATOM_B)
        for t in (0, 0.27, 1.1):
            matrix = H(t)
            self.assertTrue(matrix.isherm)
            for final, envelope in ((ATOM_R, 2 + 1j), (ATOM_R_TIL, 0.4j)):
                excited = state([0], 0, final)
                expected = envelope / 2 * np.exp(-3j * t)
                self.assertAlmostEqual(matrix.matrix_element(excited.dag(), ground), expected)
                self.assertAlmostEqual(matrix.matrix_element(ground.dag(), excited), expected.conjugate())
            photon = state([1], 0, ATOM_B)
            self.assertAlmostEqual(matrix.matrix_element(photon.dag(), ground), cavity_f(t))

    def test_solver_reproduces_resonant_pi_pulse(self):
        omega = 2.0
        H = quantumsystem().hamiltonian(
            [2], [0], [0, 0], [0, 0, omega, 3], [[0]],
            laser_drives=make_rydberg_laser(omega, lambda t: 1, lambda t: 0),
        )
        initial = state([0], 0, ATOM_B)
        target = state([0], 0, ATOM_R)
        result = qt.sesolve(H, initial, np.linspace(0, np.pi, 101),
                            e_ops=[target.proj()],
                            options={"atol": 1e-10, "rtol": 1e-10})
        np.testing.assert_allclose(result.expect[0],
                                   np.sin(np.linspace(0, np.pi, 101) / 2) ** 2,
                                   atol=1e-7)

    def test_invalid_parameters_are_rejected(self):
        defaults = dict(cutoffs=[2], wc=[5], wt=[0, 11],
                        wa=[0, 2, 17, 19], gt=[[0]])
        invalid = [
            {"cutoffs": [2.5]}, {"wa": [0, 1, 2]}, {"wt": [0, 1j]},
            {"gt": [[np.nan]]}, {"cavity_drives": [(-1, lambda t: 1)]},
            {"laser_drives": [(1, 4, lambda t: 1)]},
            {"atom_couplings": [(0, 3, 2, 1)]},
        ]
        for replacement in invalid:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                build_hamiltonian(**(defaults | replacement))


if __name__ == "__main__":
    unittest.main()
