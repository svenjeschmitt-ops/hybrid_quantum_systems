"""Fixed transmon and atomic basis, including old settings compatibility."""
import unittest
import qutip as qt

import backend as b
from hybrid_settings import FIELDS, validate


class FixedBasisTests(unittest.TestCase):
    def test_dimensions_and_controls(self):
        keys = {row[1] for row in FIELDS}
        self.assertNotIn('NC', keys)
        self.assertNotIn('NT', keys)
        self.assertNotIn('RYDBERG_MANIFOLD', keys)
        self.assertEqual((b.NT, b.NC, b.NA), (2, 2, 6))
        self.assertEqual(set(b.ATOM_STATES), {'h0', 'h1', 'r0', 'r1', 'lost', 'wrong'})
        self.assertEqual(b.H_anharm.norm(), 0)
        self.assertEqual(b.Ptransmon_leakage.norm(), 0)

    def test_legacy_settings_are_not_silently_discarded(self):
        self.assertEqual(validate({'NC': 2, 'NT': 2, 'RYDBERG_MANIFOLD': []}), validate({}))
        with self.assertRaisesRegex(ValueError, 'exactly 2 cavity states'):
            validate({'NC': 3})
        with self.assertRaisesRegex(ValueError, 'exactly 2 transmon levels'):
            validate({'NT': 3})
        with self.assertRaisesRegex(ValueError, 'Additional Rydberg levels'):
            validate({'RYDBERG_MANIFOLD': [{'label': 'extra'}]})

    def test_write_and_read_evolve_in_fixed_space(self):
        _, write = b.simulate(0, 1, noisy=False, include_storage=False)
        _, read, *_ = b.evolve_read(b.retrieval_initial(0, 1), noisy=False)
        for states in (write, read):
            self.assertEqual(states[-1].dims, [[2, 2, 6], [1]])
            self.assertAlmostEqual(states[-1].norm(), 1, places=6)
        self.assertGreaterEqual(qt.expect(b.Pwrong, write[-1]), 0)


if __name__ == '__main__':
    unittest.main()
