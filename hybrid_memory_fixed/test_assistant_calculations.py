"""Offline evidence, process cancellation, and pre-answer comparison checks."""
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from assistant_calculations import (ComparisonCancelled, checked_answer, compare_results,
                                   comparison_configs, reference_only, run_comparison)
from assistant_proposals import parse_proposal
from codex_assistant import tuning_context
from hybrid_settings import validate

MODE = 'Both transfer directions'


def proposal(changes=None):
    return dict(base_settings=validate({}),mode=MODE,changes=changes or {'PHI_SWAP':.27},
                evidence='untested trial',reason='A trial to compare.')


def point(config, score, selected=.8, lows=None):
    return dict(settings=config,mode=MODE,directions={direction:dict(
        selected_input_fidelity=selected, logical_fidelities=[.99,score,.9],
        sampled_logical_lows=lows or [.99,score,.9], tolerance_points=1)
        for direction in ('write','read')})


class ComparisonTests(unittest.TestCase):
    def test_worker_supports_every_panel_output_mode(self):
        from hybrid_panel import MODES as panel_modes
        from assistant_calculations import MODES as calculation_modes
        self.assertEqual(set(dict(panel_modes)),set(calculation_modes))

    def test_exact_resonance_calculation_is_distinct_from_performance_testing(self):
        p=proposal()
        p['evidence']='calculated model reference'
        values=tuning_context(p['base_settings'],MODE)['resonance_reference']
        p['changes']={key:values[key] for key in ('PHI_SWAP','V_SWAP')}
        self.assertTrue(reference_only(p))
        p['changes']['PHI_SWAP']+=.001
        self.assertFalse(reference_only(p))
        p['evidence']='untested trial'
        self.assertFalse(reference_only(p))

    def test_optimizing_cannot_change_noise_storage_input_or_tolerances(self):
        for changes in ({'T1_T':1000},{'STORAGE':0},{'INPUT_THETA':0},
                        {'FLUX_TOL':.01},{'NT':2},{'FC':4000}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError,'keeps the device'):
                comparison_configs(proposal(changes))

    def test_worse_excited_input_is_not_hidden_by_a_better_selected_input(self):
        p=proposal()
        old,new=comparison_configs(p)
        result=compare_results(p,point(old,.8,selected=.6),point(new,.2,selected=.9))
        self.assertFalse(result['improved'])
        answer=checked_answer(p,result)
        self.assertIn('did not improve',answer)
        self.assertNotIn('```panel-settings',answer)

    def test_improved_candidate_has_measured_values_and_a_valid_apply_proposal(self):
        p=proposal()
        old,new=comparison_configs(p)
        result=compare_results(p,point(old,.2),point(new,.8))
        answer=checked_answer(p,result)
        _,applied,error=parse_proposal(answer,old,MODE)
        self.assertIsNone(error)
        self.assertEqual(applied['changes'],p['changes'])
        self.assertEqual(applied['evidence'],'measured simulation result')
        self.assertIn('not an optimization',answer)

    def test_sampled_tolerance_floors_are_used_in_the_comparison(self):
        p=proposal()
        old,new=comparison_configs(p)
        result=compare_results(p,point(old,.8,lows=[.9,.6,.8]),point(new,.9,lows=[.9,.5,.9]))
        self.assertFalse(result['improved'])
        self.assertEqual(result['baseline_score'],.6)
        self.assertEqual(result['candidate_score'],.5)

    def test_provenance_and_nonfinite_values_are_rejected(self):
        p=proposal()
        old,new=comparison_configs(p)
        with self.assertRaisesRegex(ValueError,'does not match'):
            compare_results(p,point(new,.8),point(new,.9))
        for value in (float('nan'),float('inf'),1.1):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'Invalid'):
                compare_results(p,point(old,.8),point(new,value))

    def test_worker_results_match_exact_settings_and_are_saved(self):
        p=proposal()
        def spawn(command,**kwargs):
            config=json.loads(Path(command[command.index('--config')+1]).read_text())
            output=Path(command[command.index('--output')+1])
            score=.2 if output.parent.name=='baseline' else .8
            output.write_text(json.dumps(point(config,score)))
            self.assertEqual(kwargs['env']['MPLBACKEND'],'Agg')
            self.assertNotIn('HYBRID_STOP_FILE',kwargs['env'])
            return Mock(wait=Mock(return_value=0),poll=Mock(return_value=0))
        with tempfile.TemporaryDirectory() as folder,patch('assistant_calculations.subprocess.Popen',side_effect=spawn) as launch,patch.dict('os.environ',{'HYBRID_STOP_FILE':'old-stop'}):
            result=run_comparison(p,threading.Event(),folder)
            self.assertEqual(launch.call_count,2)
            self.assertTrue(result['improved'])
            self.assertTrue(Path(result['source']).is_file())
            self.assertEqual(json.loads(Path(result['source']).read_text())['candidate']['settings']['PHI_SWAP'],.27)

    def test_cancel_before_start_does_not_launch_a_worker(self):
        event=threading.Event()
        event.set()
        with patch('assistant_calculations.subprocess.Popen') as launch,self.assertRaises(ComparisonCancelled):
            run_comparison(proposal(),event,Path.cwd())
        launch.assert_not_called()

    def test_cancel_during_work_terminates_the_calculation(self):
        event=threading.Event()
        def wait(**kwargs):
            event.set()
            raise subprocess.TimeoutExpired('calculation',.2)
        process=Mock(wait=wait)
        with tempfile.TemporaryDirectory() as folder,patch('assistant_calculations.subprocess.Popen',return_value=process),patch('codex_assistant._stop') as stop:
            with self.assertRaises(ComparisonCancelled):
                run_comparison(proposal(),event,folder)
            stop.assert_called_once_with(process)

    def test_timeout_terminates_the_worker_without_returning_values(self):
        process=Mock()
        with tempfile.TemporaryDirectory() as folder,patch('assistant_calculations.subprocess.Popen',return_value=process),patch('assistant_calculations.time.monotonic',side_effect=[0,1201]),patch('codex_assistant._stop') as stop:
            with self.assertRaisesRegex(RuntimeError,'timed out'):
                run_comparison(proposal(),threading.Event(),folder)
            stop.assert_called_once_with(process)

    def test_failed_calculation_never_produces_an_improved_proposal(self):
        process=Mock(wait=Mock(return_value=1))
        with tempfile.TemporaryDirectory() as folder,patch('assistant_calculations.subprocess.Popen',return_value=process),patch('codex_assistant._stop') as stop:
            with self.assertRaisesRegex(RuntimeError,'Numerical comparison failed'):
                run_comparison(proposal(),threading.Event(),folder)
            stop.assert_called_once_with(process)


class PreAnswerTests(unittest.TestCase):
    def panel(self):
        import queue
        from hybrid_panel import ParameterPanel
        panel=ParameterPanel.__new__(ParameterPanel)
        panel.assistant_request=0
        panel.assistant_history=[]
        panel.assistant_proposal=None
        panel.assistant_preview=None
        panel.last_run=None
        panel.process=None
        panel.events=queue.Queue()
        panel.backend_path=Path.cwd()/'hybrid_memory_fixed'/'backend.py'
        panel.config=Mock(return_value=validate({}))
        for name in ('assistant_question','mode','test_suggestions','ask_button',
                     'cancel_answer_button','apply_values_button','assistant_status',
                     'assistant_append','root'):
            setattr(panel,name,Mock())
        panel.ask_button.instate.return_value=False
        panel.assistant_question.get.return_value='Improve the transfer'
        panel.mode.get.return_value=MODE
        panel.test_suggestions.get.return_value=True
        return panel

    def ask(self,panel,answer,comparison=None):
        def launch(**kwargs):
            return Mock(start=kwargs['target'])
        def calculate(*args,**kwargs):
            kwargs['on_progress']('Testing proposed values...')
            return comparison
        with patch('hybrid_panel.threading.Thread',side_effect=launch),patch('hybrid_panel.codex_answer',return_value=iter([answer])),patch('hybrid_panel.run_comparison',side_effect=calculate) as check:
            panel.ask_assistant()
            panel.poll()
        return check

    def test_numerical_check_happens_before_a_trial_proposal_can_be_applied(self):
        for improved in (False,True):
            with self.subTest(improved=improved):
                panel=self.panel()
                p=proposal()
                old,new=comparison_configs(p)
                comparison=compare_results(p,point(old,.5),point(new,.8 if improved else .2))
                answer='An unverified guess.\n```panel-settings\n'+json.dumps({
                    'changes':p['changes'],'evidence':'untested trial','reason':'A trial.'})+'\n```'
                check=self.ask(panel,answer,comparison)
                check.assert_called_once()
                self.assertEqual(panel.config(),old)
                self.assertEqual(panel.assistant_history[-1]['numerical_comparison'],comparison)
                self.assertNotIn('An unverified guess',panel.assistant_history[-1]['answer'])
                if improved:
                    self.assertEqual(panel.assistant_proposal['evidence'],'measured simulation result')
                    panel.apply_values_button.configure.assert_called_with(state='normal')
                else:
                    self.assertIsNone(panel.assistant_proposal)
                    panel.apply_values_button.configure.assert_called_with(state='disabled')

    def test_formula_resonance_reference_does_not_launch_a_performance_test(self):
        panel=self.panel()
        value=tuning_context(validate({}),MODE)['resonance_reference']['PHI_SWAP']
        answer='Bare resonance reference only.\n```panel-settings\n'+json.dumps({
            'changes':{'PHI_SWAP':value},'evidence':'calculated model reference','reason':'Restore bare resonance.'})+'\n```'
        check=self.ask(panel,answer)
        check.assert_not_called()
        self.assertEqual(panel.assistant_proposal['evidence'],'calculated model reference')

    def test_disabling_trial_tests_preserves_explicit_untested_status(self):
        panel=self.panel()
        panel.test_suggestions.get.return_value=False
        answer='An untested comparison.\n```panel-settings\n'+json.dumps({
            'changes':{'PHI_SWAP':.27},'evidence':'untested trial','reason':'A trial.'})+'\n```'
        check=self.ask(panel,answer)
        check.assert_not_called()
        self.assertEqual(panel.assistant_proposal['evidence'],'untested trial')


if __name__=='__main__':
    unittest.main()
