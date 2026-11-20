import tempfile
import unittest

from assistant_search import requested_tc_coupling_bounds, search_space, optimize_controls, validate_search_result, search_answer
from assistant_proposals import parse_proposal
from hybrid_settings import validate, coupling_strengths

MODE = 'Both transfer directions'


def point(config, score):
    return {'settings': config, 'mode': MODE, 'directions': {name: {
        'sampled_logical_lows': [score]*3, 'logical_fidelities': [score]*3,
        'selected_input_fidelity': score} for name in ('write', 'read')}}


class CouplingSearchTests(unittest.TestCase):
    def test_extract_range_from_user_request(self):
        self.assertEqual(requested_tc_coupling_bounds('I can adjust the couling strenght of the transmon-cavity between 0.5MHz to 50 MHz'), [.5, 50.])
        self.assertIsNone(requested_tc_coupling_bounds('set the cavity frequency between 1 and 50 MHz'))
        with self.assertRaises(ValueError):
            requested_tc_coupling_bounds('transmon cavity coupling between 50 MHz and 0.5 MHz')

    def test_searches_manual_coupling_and_preserves_atom_coupling(self):
        for manual in (0, 1):
            c = validate({'DIRECT_COUPLING': manual, 'FC': 6120.})
            keys, _, _, seeds, decode = search_space(c, MODE, False, [.5, 50.])
            self.assertIn('G_TC_MHZ', keys)
            values = [decode(seed)['G_TC_MHZ'] for seed in seeds]
            self.assertTrue(any(abs(v-.5) < 1e-8 for v in values))
            self.assertTrue(any(abs(v-50) < 1e-8 for v in values))
            candidate = decode(seeds[-1])
            self.assertEqual(candidate['DIRECT_COUPLING'], 1)
            self.assertAlmostEqual(coupling_strengths(candidate)['atom_coupling_MHz'], coupling_strengths(c)['atom_coupling_MHz'])
            result = {'mode': MODE, 'target': .8, 'vary_cavity': False,
                      'tc_coupling_bounds': [.5, 50.], 'baseline': point(c, .01),
                      'best': point(candidate, .4), 'searched_parameters': keys}
            verified = validate_search_result(result, c, MODE, .8, False, [.5, 50.])
            self.assertIn('G_TC_MHZ', verified['changes'])
            _, proposal, error = parse_proposal(search_answer(result, c, MODE, .8), c, MODE)
            self.assertIsNone(error)
            self.assertIn('G_TC_MHZ', proposal['changes'])
            with self.assertRaises(ValueError):
                validate_search_result(result, c, MODE, .8, False, [1., 50.])

    def test_optimizer_evaluates_multiple_couplings(self):
        c = validate({'DIRECT_COUPLING': 1, 'FC': 6120.})
        with tempfile.TemporaryDirectory() as folder:
            result = optimize_controls(c, MODE, .8, folder, max_evaluations=15,
                evaluator=lambda candidate, stage: 0., tc_coupling_bounds=[.5, 50.])
        values = {round(row['settings']['G_TC_MHZ'], 3) for row in result['trace'] if 'settings' in row}
        self.assertIn(.5, values)
        self.assertIn(50., values)


if __name__ == '__main__':
    unittest.main()
