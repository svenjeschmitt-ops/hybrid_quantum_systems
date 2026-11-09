"""Numerical report provenance, bounded previews and assistant handoff."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from codex_assistant import build_messages
from hybrid_settings import validate
from simulation_output import load_reports, trace, trace_preview, write_report


class NumericalReportTests(unittest.TestCase):
    def publish(self, folder, direction="write", settings=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return write_report(folder,direction,settings or validate({}),
                {"time_origin":"before storage","stages_us":{"swap":[0.,1.]}},
                {"fidelity":np.float64(.8)},
                {"dynamics":trace([0.,.5,1.],joint_fidelity=[0.,.3,.8],cavity_mean_photons=[0.,.7,.1])},
                {"input_labels":["g","e","+"],"joint_fidelities":np.array([.9,.8,.85])})

    def test_full_resolution_artifact_and_preview_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.publish(folder)
            full=json.loads(path.read_text())
            self.assertEqual(full["traces"]["dynamics"]["series"]["joint_fidelity"],[0.,.3,.8])
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            reports,errors=load_reports(folder,validate({}))
            self.assertFalse(errors)
            self.assertEqual(reports[0]["full_resolution_file"],str(path.resolve()))
            self.assertEqual(reports[0]["summary"]["fidelity"],.8)

    def test_extrema_survive_downsampled_preview(self):
        values=np.zeros(1001)
        values[17]=.95
        preview=trace_preview(trace(np.arange(1001)/100,wrong_rydberg=values))
        self.assertLessEqual(len(preview["sampled_time_us"]),33)
        stats=preview["exact_sampled_grid_statistics"]["wrong_rydberg"]
        self.assertEqual(stats["maximum"],.95)
        self.assertEqual(stats["maximum_time_us"],.17)
        self.assertEqual(preview["full_point_count"],1001)
        self.assertNotIn(.95,preview["sampled_series"]["wrong_rydberg"])

    def test_wrong_settings_rejected_not_attached(self):
        with tempfile.TemporaryDirectory() as folder:
            self.publish(folder,settings=validate({"FC":4900}))
            reports,errors=load_reports(folder,validate({}))
            self.assertFalse(reports)
            self.assertIn("do not match",errors[0])

    def test_partial_run_context_keeps_reports_and_current_edits_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            settings=validate({})
            self.publish(folder)
            run={"settings":settings,"status":"running","output":"partial log","output_directory":folder}
            messages=build_messages("Explain leakage",{"FC":4900},"Both transfer directions",run)
            context=json.loads(messages[1]["content"].split("\n",1)[1].split("\nQuestion:")[0])
            self.assertEqual(context["current_panel"]["settings"]["FC"],4900)
            previous=context["last_simulation"]
            self.assertEqual(previous["settings"]["FC"],5000)
            self.assertEqual(previous["status"],"running")
            self.assertEqual([report["direction"] for report in previous["numerical_reports"]],["write"])
            self.assertNotIn("numerical_reports",run)

    def test_bad_file_does_not_hide_available_other_direction(self):
        with tempfile.TemporaryDirectory() as folder:
            self.publish(folder)
            (Path(folder)/"read.json").write_text('{"partial":')
            reports,errors=load_reports(folder,validate({}))
            self.assertEqual([r["direction"] for r in reports],["write"])
            self.assertEqual(len(errors),1)

    def test_trace_rejects_mismatched_or_invalid_data(self):
        for times,values in (([],[]),([0.,1.],[.5]),([1.,0.],[0.,1.]),([0.,1.],[0.,np.nan])):
            with self.subTest(times=times,values=values),self.assertRaises(ValueError):
                trace(times,value=values)

    def test_missing_reports_explicitly_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_reports(folder,validate({})),([],[]))


if __name__=="__main__":
    unittest.main()
