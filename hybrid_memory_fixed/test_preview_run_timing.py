"""Preview must reflect the selected timings in a matching run report."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from hybrid_settings import validate
from pulse_preview import control_preview
from pulse_preview_panel import PulsePreview


class PreviewRunTimingTests(unittest.TestCase):
    def test_report_selection_only_uses_matching_current_run(self):
        settings = validate({'CONSTANT_FLUX': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'read.json'
            path.write_text(json.dumps({'metadata': {}, 'direction':'read'}))
            preview = PulsePreview.__new__(PulsePreview)
            preview.get_settings = lambda: settings
            preview.get_run = lambda: {'settings': settings, 'output_directory': directory}
            preview.direction = Mock()
            preview.schedule = Mock()
            preview.report_ready('read',select=True)
            preview.direction.set.assert_called_once_with('Read')
            preview.schedule.assert_called_once()
            preview.direction.reset_mock(); preview.schedule.reset_mock()
            preview.get_settings = lambda: {**settings,'FC':settings['FC']+1}
            preview.report_ready('read',select=True)
            preview.direction.set.assert_not_called()
            preview.schedule.assert_not_called()

    def test_write_and_read_use_reported_stages_and_control_traces(self):
        settings = validate({'CONSTANT_FLUX': 1})
        times = [0., .02, .06, .12, .14]
        series = dict(flux_Phi_over_Phi0=[.26]*5,
                      electrode_voltage_V=[-.02, -.02, .39, -.02, -.02],
                      transmon_detuning_MHz=[-54.]*5,
                      atom_detuning_MHz=[150., 150., -20., 150., 150.])
        for direction in ('write', 'read'):
            with self.subTest(direction=direction):
                stages = (dict(preparation=[0.,.01], transmon_cavity=[.02,.25],
                               atom_cavity=[.025,.11], mapping=[.13,.14]) if direction == 'write' else
                          {'Memory to Rydberg':[0.,.01], 'Cavity to transmon':[.115,.345],
                           'Atom to cavity':[.02,.105], 'Return atom to h0':[.13,.14]})
                report = dict(schema_version=1,direction=direction,settings=settings,
                              metadata={'stages_us':stages},
                              summary={'idealized_coupling_switch':{'time_us':.025}},
                              traces={'controls':{'time_us':times,'series':series}})
                view = control_preview(settings,direction,report=report)
                self.assertEqual(view['source'],'simulation')
                self.assertEqual(view['stages']['mapping'],(.13,.14))
                self.assertEqual(view['time_us'],times)
                self.assertEqual(view['voltage'],series['electrode_voltage_V'])
                self.assertEqual(view['coupling_switch_us'],.025)
                self.assertAlmostEqual(view['finish_us'],.14)
                self.assertTrue(np.isfinite(view['optical0']).all())
                with self.assertRaises(ValueError):
                    control_preview({**settings, 'FC': settings['FC']+1},direction,report=report)
                with self.assertRaises(ValueError):
                    control_preview(settings, 'read' if direction == 'write' else 'write',report=report)


if __name__ == '__main__':
    unittest.main()
