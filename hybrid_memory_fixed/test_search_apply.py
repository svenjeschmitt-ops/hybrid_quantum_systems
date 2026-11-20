import unittest
from unittest.mock import Mock
from types import SimpleNamespace

from assistant_search import search_answer
from hybrid_panel import ParameterPanel
from hybrid_settings import validate

MODE = 'Both transfer directions'


def point(settings, score):
    return {'settings': settings, 'mode': MODE, 'directions': {
        direction: {'sampled_logical_lows': [score] * 3,
                    'logical_fidelities': [score] * 3,
                    'selected_input_fidelity': score}
        for direction in ('write', 'read')}}


class ApplySearchTests(unittest.TestCase):
    def test_below_target_result_is_available_for_preview(self):
        baseline = validate({})
        candidate = validate({**baseline, 'PHI_SWAP': .25})
        result = {'mode': MODE, 'target': .8, 'vary_cavity': False,
                  'baseline': point(baseline, .01), 'best': point(candidate, .4),
                  'searched_parameters': ['PHI_SWAP']}
        answer = search_answer(result, baseline, MODE, .8)
        self.assertIn('Target not reached', answer)
        self.assertIn('below the 80.00% target', answer)
        panel = SimpleNamespace(assistant_pending={'settings': baseline, 'mode': MODE,
                                'fidelity_target': .8, 'target_search': result},
                                assistant_history=[], assistant_proposal=None,
                                apply_values_button=Mock(), assistant_append=Mock())
        ParameterPanel.receive_assistant_answer(panel, answer)
        self.assertEqual(panel.assistant_proposal['changes'], {'PHI_SWAP': .25})
        panel.apply_values_button.configure.assert_called_with(state='normal')

    def test_unverified_changes_are_rejected(self):
        baseline = validate({})
        candidate = validate({**baseline, 'PHI_SWAP': .25})
        result = {'mode': MODE, 'target': .8, 'vary_cavity': False,
                  'baseline': point(baseline, .01), 'best': point(candidate, .4),
                  'searched_parameters': ['PHI_SWAP']}
        answer = search_answer(result, baseline, MODE, .8).replace('0.25}', '0.26}')
        panel = SimpleNamespace(assistant_pending={'settings': baseline, 'mode': MODE,
                                'fidelity_target': .8, 'target_search': result},
                                assistant_history=[], assistant_proposal=None,
                                apply_values_button=Mock(), assistant_append=Mock())
        ParameterPanel.receive_assistant_answer(panel, answer)
        self.assertIsNone(panel.assistant_proposal)
        panel.apply_values_button.configure.assert_called_with(state='disabled')


if __name__ == '__main__':
    unittest.main()
