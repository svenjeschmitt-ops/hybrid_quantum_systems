"""Sweep grid validation and fresh-model/export integration checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hybrid_settings import DEFAULTS
from parameter_sweep import sweep_configs, run


class SweepGridTests(unittest.TestCase):
    def test_inclusive_and_descending_grid_without_baseline_mutation(self):
        config = dict(DEFAULTS)
        points = sweep_configs(config, 'PHI_SWAP', .26, .27, .001)
        self.assertEqual(len(points), 11)
        self.assertEqual(points[-1]['PHI_SWAP'], .27)
        self.assertEqual(config, DEFAULTS)
        reverse = sweep_configs(config, 'PHI_SWAP', .27, .26, -.001)
        self.assertEqual(len(reverse), 11)
        self.assertEqual(reverse[-1]['PHI_SWAP'], .26)

    def test_off_grid_stop_is_not_added(self):
        points = sweep_configs({}, 'STORAGE', 0, 1, .3)
        self.assertEqual(len(points), 4)
        self.assertAlmostEqual(points[-1]['STORAGE'], .9)

    def test_invalid_grids_and_invalid_interior_points_fail_before_running(self):
        for start, stop, step in [(0, 1, 0), (0, 1, -.1), (1, 0, .1),
                                  (0, 0, 1), (0, 1, 2), (0, 1, .0001),
                                  (0, float('nan'), .1), (0, 1, float('inf'))]:
            with self.subTest(values=(start, stop, step)), self.assertRaises(ValueError):
                sweep_configs({}, 'STORAGE', start, stop, step)
        with self.assertRaisesRegex(ValueError, 'Sweep point 2'):
            sweep_configs({}, 'DIPOLE_ALIGNMENT', -1, 1, 1)
        with self.assertRaises(ValueError):
            sweep_configs({}, 'PHI_SWAP', .49, .51, .01)

    def test_run_writes_completed_points_and_passes_fresh_settings(self):
        from types import SimpleNamespace
        backend = SimpleNamespace(__file__=str(Path(__file__).with_name('backend.py')))
        seen = []
        def fake_solver(command, **kwargs):
            config = json.loads(Path(command[command.index('--config')+1]).read_text())
            seen.append(config)
            output = Path(command[command.index('--sweep-point')+1])
            output.write_text(json.dumps({'directions': {
                'write': {'final_fidelity': config['PHI_SWAP'], 'fidelity_range': {'low': .1, 'high': .9}},
                'read': {'final_fidelity': .5}}}))
        with tempfile.TemporaryDirectory() as directory, patch('parameter_sweep.subprocess.run', side_effect=fake_solver), patch('sweep_figure.SweepFigure'):
            report = run(backend, {**DEFAULTS, 'FLUX_TOL': .001}, 'PHI_SWAP', .26, .27, .01, output_directory=directory)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['tolerance_points_per_sweep_point'], 3)
            self.assertTrue(all(c['FLUX_TOL'] == .001 for c in seen))
            self.assertEqual([c['PHI_SWAP'] for c in seen], [.26, .27])
            with (Path(directory)/'parameter_sweep.csv').open() as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[1]['write_fidelity']), .27)
            self.assertEqual(float(rows[1]['write_fidelity_low']), .1)
            self.assertEqual(float(rows[1]['write_fidelity_high']), .9)
            self.assertEqual(float(rows[1]['read_fidelity_low']), .5)
            self.assertTrue((Path(directory)/'point_0001.json').is_file())
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_failed_worker_leaves_incomplete_report(self):
        from types import SimpleNamespace
        backend = SimpleNamespace(__file__=str(Path(__file__).with_name('backend.py')))
        with tempfile.TemporaryDirectory() as directory, patch('parameter_sweep.subprocess.run', side_effect=RuntimeError('solver failed')):
            with self.assertRaisesRegex(RuntimeError, 'solver failed'):
                run(backend, DEFAULTS, 'PHI_SWAP', .26, .27, .01, output_directory=directory)
            report = json.loads((Path(directory)/'parameter_sweep.json').read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['results'], [])


    def test_figure_slider_updates_every_panel_and_supports_descending_values(self):
        import numpy as np
        import backend
        from sweep_figure import SweepFigure
        controls = {'time_us': [0, 1], 'series': {
            'flux_Phi_over_Phi0': [0, .25], 'electrode_voltage_V': [0, .4],
            'transmon_detuning_MHz': [800, -50], 'atom_detuning_MHz': [150, -20]}}
        names = ('transmon_e', 'cavity_mean_photons', 'atom_h0', 'atom_h1',
                 'atom_r0', 'atom_r1', 'atom_loss', 'atom_wrong_rydberg')
        with tempfile.TemporaryDirectory() as directory:
            rows=[]
            for index, value in enumerate((.27, .26)):
                payload = {'metadata': {'stages_us': {'swap': [0, 1]},
                           'input_theta_degrees': 90, 'input_phi_degrees': 60, 'storage_duration_us': 5},
                           'traces': {'controls': controls, 'selected_input_dynamics': {
                               'time_us': [0, 1], 'series': {name: [0, value] for name in names}}},
                           'summary': {'displayed_metrics': [['Fidelity', str(value), '—']]},
                           'basis_outputs': {'input_labels': ['g → h0', 'e → h1', '+ → +'],
                                             'probabilities': [[1, 0, 0, 0], [0, 1, 0, 0], [.5, .5, 0, 0]]}}
                # Distinct controls, table and basis probabilities at the next point.
                payload['traces']['controls'] = {'time_us': [0, 1], 'series': {
                    **controls['series'], 'flux_Phi_over_Phi0': [0, value]}}
                payload['basis_outputs']['probabilities'][0] = [value, 0, 0, 1-value]
                path = Path(directory)/f'point_{index}.json'
                path.write_text(json.dumps({'directions': {'write': {'figure_report': payload}}}))
                rows.append({'PHI_SWAP': value, 'data_file': path.name})
            viewer = SweepFigure(backend, {'parameter': 'PHI_SWAP', 'unit': 'Phi/Phi0', 'results': rows}, directory, 'write')
            self.assertEqual(viewer.index, 0)
            viewer.slider.set_val(1)
            self.assertEqual(viewer.index, 1)
            self.assertEqual(viewer.voltage_axis.yaxis.get_label_position(), 'right')
            self.assertIn('0.26', viewer.heading.get_text())
            np.testing.assert_allclose(viewer.control_axis.lines[0].get_ydata(), [0, .26])
            np.testing.assert_allclose(viewer.population_axis.lines[0].get_ydata(), [0, .26])
            self.assertEqual(viewer.summary_axis.tables[0][1, 1].get_text().get_text(), '0.26')
            self.assertEqual(len(viewer.basis_axis.collections), 9)
            viewer.move(-1)
            self.assertEqual(viewer.index, 0)
            viewer.move(-1)
            self.assertEqual(viewer.index, 0)
            # Only point 2 has tolerance ranges: dragging must add and remove
            # bands and the range column, without accumulating collections.
            path = Path(directory)/'point_1.json'
            point = json.loads(path.read_text())
            payload = point['directions']['write']['figure_report']
            payload['summary']['tolerance_ranges'] = {'point_count': 3}
            payload['summary']['displayed_metrics'][0][2] = '0.25–0.27'
            payload['traces']['selected_input_tolerance_ranges'] = {
                'time_us': [0, 1], 'series': {name+'_'+side: [0, .25 if side=='low' else .27]
                    for name in (*names, 'flux', 'transmon_detuning', 'atom_detuning')
                    for side in ('low', 'high')}}
            point['directions']['read'] = point['directions']['write']
            path.write_text(json.dumps(point))
            viewer.slider.set_val(1)
            self.assertEqual(len(viewer.population_axis.collections), 8)
            self.assertEqual(len(viewer.detuning_axis.collections), 2)
            self.assertEqual(len(viewer.control_axis.collections), 1)
            self.assertEqual(viewer.summary_axis.tables[0][1, 2].get_text().get_text(), '0.25–0.27')
            self.assertIn('3 fixed-error', viewer.footer.get_text())
            viewer.slider.set_val(0)
            self.assertEqual(len(viewer.population_axis.collections), 0)
            viewer.slider.set_val(1)
            self.assertEqual(len(viewer.population_axis.collections), 8)
            import matplotlib.pyplot as plt
            plt.close(viewer.figure)
            read_report = {'parameter': 'PHI_SWAP', 'unit': 'Phi/Phi0', 'results': [rows[1], rows[1]]}
            read_viewer = SweepFigure(backend, read_report, directory, 'read')
            self.assertEqual(len(read_viewer.population_axis.collections), 8)
            self.assertIn('With tolerances', read_viewer.summary_axis.tables[0][0, 2].get_text().get_text())
            plt.close(read_viewer.figure)



    def test_point_evaluation_keeps_tolerances_and_exports_fidelity_bounds(self):
        from types import SimpleNamespace
        from parameter_sweep import evaluate_point
        from tolerance_bands import POPULATION_NAMES
        config = {**DEFAULTS, 'FLUX_TOL': .001}
        backend = SimpleNamespace(_SETTINGS=config, _ASSISTANT_OUTPUT='original')
        def fake_main(save=False):
            self.assertIs(backend._SETTINGS, config)
            self.assertEqual(backend._SETTINGS['FLUX_TOL'], .001)
            payload = {'metadata': {'phase_correction_rad': .2},
                       'summary': {'selected_fidelity_after_storage': .8,
                                   'tolerance_ranges': {'metric_ranges': {'final_fidelity': {'low': .7, 'high': .9}}}},
                       'traces': {'selected_input_dynamics': {'time_us': [0],
                           'series': {**{key: [0] for key in POPULATION_NAMES}, 'joint_fidelity': [.8]}}}}
            (Path(backend._ASSISTANT_OUTPUT)/'write.json').write_text(json.dumps(payload))
        backend.main = fake_main
        result = evaluate_point(backend, ('write',))
        self.assertEqual(result['directions']['write']['fidelity_range'], {'low': .7, 'high': .9})
        self.assertIs(backend._SETTINGS, config)
        self.assertEqual(backend._ASSISTANT_OUTPUT, 'original')


if __name__ == '__main__':
    unittest.main()
