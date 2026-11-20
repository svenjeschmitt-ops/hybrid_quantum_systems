import unittest

from assistant_search import search_space, optimize_controls
from atomic_model import AtomicModel
from hybrid_settings import calculated_swap_field
import tempfile
from hybrid_settings import DEFAULTS, validate


class RydbergSearchTests(unittest.TestCase):
    def test_unreachable_atom_searches_splitting_and_field(self):
        config = validate({**DEFAULTS, 'FC': 6120.})
        for vary_cavity in (False, True):
            keys, bounds, current, seeds, decode = search_space(config, 'Both transfer directions', vary_cavity)
            self.assertIn('F_A_ZERO', keys)
            self.assertIn('V_SWAP', keys)
            candidate = decode(seeds[-1])
            self.assertGreater(candidate['F_A_ZERO'], config['F_A_ZERO'])
            if not vary_cavity:
                self.assertEqual(candidate['FC'], config['FC'])
            self.assertEqual(len(bounds), len(current))

    def test_joint_seeds_explore_distinct_fields_at_resonance(self):
        config = validate({**DEFAULTS, 'FC': 6120.})
        keys, _, _, seeds, decode = search_space(config, 'Both transfer directions')
        points = [decode(seed) for seed in seeds[1:]]
        self.assertGreaterEqual(len({round(point['V_SWAP'], 6) for point in points}), 3)
        for point in points:
            frequency = float(AtomicModel(point).transition(calculated_swap_field(point)))
            self.assertAlmostEqual(frequency, point['FC'])
        with tempfile.TemporaryDirectory() as folder:
            result = optimize_controls(config, 'Both transfer directions', .8, folder,
                                       max_evaluations=12, evaluator=lambda c, stage: 0.)
        fields = {round(row['settings']['V_SWAP'], 6) for row in result['trace'] if 'settings' in row}
        self.assertGreaterEqual(len(fields), 3)

    def test_locked_field_is_preserved(self):
        config = validate({**DEFAULTS, 'FC': 6120., 'CONSTANT_FIELD': 1})
        keys, _, _, seeds, decode = search_space(config, 'Both transfer directions')
        self.assertNotIn('V_SWAP', keys)
        self.assertEqual(decode(seeds[-1])['V_SWAP'], config['V_SWAP'])


if __name__ == '__main__':
    unittest.main()
