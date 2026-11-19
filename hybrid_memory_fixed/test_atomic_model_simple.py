"""Checks for the simplified atomic model and legacy table handling."""
import json
import unittest

import numpy as np

from atomic_model import AtomicModel
from hybrid_settings import resonant_field, validate


class SimplifiedAtomicModelTests(unittest.TestCase):
    def setUp(self):
        self.config = validate({})
        self.atom = AtomicModel(self.config)

    def test_quadratic_transition_and_resonance(self):
        fields = np.array([0., 1., 3.])
        c = self.config
        expected = c['F_A_ZERO'] + (c['K_R1'] - c['K_R0'])*fields**2
        np.testing.assert_allclose(self.atom.transition(fields), expected)
        self.assertAlmostEqual(float(self.atom.transition(resonant_field(c))), c['FC'])

    def test_coupling_is_field_independent(self):
        values = self.atom.coupling('r1<-r0', np.array([0., 1., 3.]))
        np.testing.assert_allclose(values, np.full(3, values[0]))

    def test_old_empty_table_loads_and_nonempty_table_is_rejected(self):
        self.assertEqual(validate({'ATOMIC_SPECTRUM': '{}'}), self.config)
        with self.assertRaisesRegex(ValueError, 'ATOMIC_SPECTRUM has been removed'):
            validate({'ATOMIC_SPECTRUM': json.dumps({'energies_MHz': {'r0': [0]}})})


if __name__ == '__main__':
    unittest.main()
