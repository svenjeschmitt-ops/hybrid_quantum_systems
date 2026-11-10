"""Progress ranges, subprocess mapping and failure semantics."""
import contextlib
import io
import unittest
from unittest.mock import patch

import simulation_progress as progress


class ProgressTests(unittest.TestCase):
    def test_nested_stages_map_to_global_percent_and_restore(self):
        output=io.StringIO()
        with patch.object(progress,'_enabled',True), patch.object(progress,'_base',0), patch.object(progress,'_span',1), contextlib.redirect_stdout(output):
            with progress.scope(.5,1,'Read'):
                with progress.scope(.2,.6,'Inputs'):
                    progress.update(.5,'Half input batch')
                env=progress.child_environment(.25,.75)
            self.assertEqual(progress._base,0)
            self.assertEqual(progress._span,1)
            progress.update(1,'Simulation ready',complete=True)
        events=[progress.parse_line(line) for line in output.getvalue().splitlines()]
        self.assertAlmostEqual(next(e['percent'] for e in events if e['label']=='Half input batch'),70)
        self.assertEqual(events[-1]['percent'],100)
        self.assertTrue(all(event['percent']<100 for event in events[:-1]))
        self.assertAlmostEqual(float(env['HYBRID_PROGRESS_START']),.625)
        self.assertAlmostEqual(float(env['HYBRID_PROGRESS_SPAN']),.25)
        self.assertEqual(env['HYBRID_SIM_PROGRESS'],'1')

    def test_failed_stage_does_not_emit_completion(self):
        output=io.StringIO()
        with patch.object(progress,'_enabled',True), patch.object(progress,'_base',0), patch.object(progress,'_span',1), contextlib.redirect_stdout(output):
            with self.assertRaises(RuntimeError):
                with progress.scope(.2,.8,'Failing solver'):
                    progress.update(.5,'Working')
                    raise RuntimeError('failed')
            self.assertEqual(progress._base,0)
        events=[progress.parse_line(line) for line in output.getvalue().splitlines()]
        self.assertEqual([e['percent'] for e in events],[20,50])

    def test_invalid_events_do_not_change_progress(self):
        for line in ('ordinary output', 'HYBRID_PROGRESS invalid',
                     'HYBRID_PROGRESS {"percent":101,"label":"x"}',
                     'HYBRID_PROGRESS {"percent":NaN,"label":"x"}',
                     'HYBRID_PROGRESS {"percent":5,"label":null}',
                     'HYBRID_PROGRESS {}'):
            self.assertIsNone(progress.parse_line(line))

    def test_tolerance_grids_receive_proportional_work_budget(self):
        nominal=progress.transfer_stages({})
        uncertain=progress.transfer_stages({'FLUX_TOL':.001})
        self.assertAlmostEqual(nominal['basis'],.95)
        self.assertAlmostEqual(uncertain['basis'],.95*6/(6+4*2))
        self.assertLess(uncertain['basis'],nominal['basis'])
        self.assertLess(uncertain['calibration'],uncertain['selected'])

    def test_disabled_reporting_is_silent(self):
        output=io.StringIO()
        with patch.object(progress,'_enabled',False), contextlib.redirect_stdout(output):
            with progress.scope(0,1,'Hidden'):
                progress.update(.5,'Hidden')
        self.assertEqual(output.getvalue(),'')


if __name__=='__main__':
    unittest.main()
