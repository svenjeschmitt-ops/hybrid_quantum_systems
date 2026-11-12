"""App-controlled numerical comparisons before presenting assistant trial values.

The language model supplies data only. A fixed local worker runs the existing
simulation, with unchanged storage, noise, basis, solver and tolerance settings.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from assistant_proposals import prepare_application
from hybrid_settings import validate

MODES = {
    'Both transfer directions': ('write', 'read'),
    'Transmon to atom': ('write',),
    'Atom to transmon': ('read',),
    'Simultaneous cavity resonance: both directions': ('write', 'read'),
    'Simultaneous cavity resonance: transmon to atom': ('write',),
    'Simultaneous cavity resonance: atom to transmon': ('read',),
}
# These alter the protocol rather than the requested memory task or noise model.
CONTROL_FIELDS = {'PHI_SWAP', 'PHI_PARK', 'V_SWAP', 'E_PARK', 'CONSTANT_FLUX',
                  'CONSTANT_FIELD', 'TC_PULSE_SCALE', 'AC_PULSE_SCALE', 'RAMP',
                  'SIM_HOLD_SCALE', 'SIM_RAMP', 'OMEGA', 'GAP'}
GAIN_THRESHOLD = 1e-5


class ComparisonCancelled(Exception):
    pass


def reference_only(proposal):
    if proposal['evidence'] != 'calculated model reference':
        return False
    from codex_assistant import tuning_context
    references = tuning_context(proposal['base_settings'], proposal['mode'])['resonance_reference']
    return all(key in ('PHI_SWAP', 'V_SWAP') and key in references
               and math.isclose(float(value), references[key], rel_tol=0, abs_tol=1e-12)
               for key, value in proposal['changes'].items())


def comparison_configs(proposal):
    changes = prepare_application(proposal, proposal['base_settings'], proposal['mode'])
    unsupported = set(changes)-CONTROL_FIELDS
    if unsupported:
        raise ValueError('Automatic comparison keeps the device, input, storage, noise, basis and tolerances fixed. '
                         'Unsupported changes: '+', '.join(sorted(unsupported)))
    if proposal['mode'] not in MODES:
        raise ValueError('This output mode is not supported for numerical comparison.')
    return validate(proposal['base_settings']), validate({**proposal['base_settings'], **changes})


def evaluate_backend(b, mode, folder):
    """Use the ordinary figure calculations so the reported fidelities agree."""
    import matplotlib.pyplot as plt
    simultaneous = mode.startswith('Simultaneous cavity resonance:')
    b._ASSISTANT_OUTPUT = str(folder)
    result = {'settings': b._SETTINGS, 'mode': mode, 'directions': {}}
    for direction in MODES[mode]:
        if simultaneous:
            from simultaneous_coupling import run
            run(b, direction=direction)
        elif direction == 'write':
            b.main()
        else:
            b.retrieval_main()
        report = json.loads((Path(folder)/(direction+'.json')).read_text())
        summary = report['summary']
        basis = ([row['joint_fidelity_final'] for row in report['basis_outputs']] if simultaneous
                 else report['basis_outputs']['joint_fidelities'])
        selected = (summary['joint_fidelity_after_storage'] if simultaneous else summary['selected_fidelity_after_storage']) if direction == 'write' else summary['selected_joint_fidelity']
        ranges = summary.get('tolerance_ranges')
        lows = [ranges['metric_ranges'][key]['low'] for key in ('basis_0', 'basis_1', 'basis_plus')] if ranges else list(basis)
        result['directions'][direction] = dict(
            selected_input_fidelity=float(selected), logical_fidelities=list(map(float, basis)),
            sampled_logical_lows=list(map(float, lows)), tolerance_points=ranges['point_count'] if ranges else 1,
            source=str((Path(folder)/(direction+'.json')).resolve()))
        plt.close('all')
    return result


def _score(point, mode):
    scores = []
    for direction in MODES[mode]:
        data = point['directions'][direction]
        for key in ('logical_fidelities', 'sampled_logical_lows'):
            values = data[key]
            if len(values) != 3 or not all(math.isfinite(v) and -1e-6 <= v <= 1+1e-6 for v in values):
                raise ValueError('Invalid comparison fidelity data.')
        if not math.isfinite(data['selected_input_fidelity']) or not -1e-6 <= data['selected_input_fidelity'] <= 1+1e-6:
            raise ValueError('Invalid selected-input fidelity.')
        scores.extend(data['sampled_logical_lows'])
    return min(scores)


def compare_results(proposal, baseline, candidate):
    before, after = comparison_configs(proposal)
    for result, expected in ((baseline, before), (candidate, after)):
        if result['settings'] != expected or result['mode'] != proposal['mode']:
            raise ValueError('Comparison output does not match the requested settings and mode.')
    old, new = _score(baseline, proposal['mode']), _score(candidate, proposal['mode'])
    return dict(status='complete', mode=proposal['mode'], changes=proposal['changes'],
                baseline=baseline, candidate=candidate, baseline_score=old, candidate_score=new,
                gain=new-old, improved=new > old+GAIN_THRESHOLD,
                objective='Minimum joint fidelity over logical 0, 1 and + inputs across the selected directions; '
                          'minimum sampled tolerance fidelities when bounds are nonzero.',
                limitation='Two configurations compared, not an optimization or proof for all qubit inputs. '
                           'Each configuration uses its own single noiseless phase calibration per direction.')


def run_comparison(proposal, cancelled, project_root, on_progress=None, timeout=1200):
    """Cancellable isolated workers; no parameter writes and no plot windows."""
    from codex_assistant import _stop
    baseline, candidate = comparison_configs(proposal)
    if cancelled.is_set():
        raise ComparisonCancelled()
    root = Path(project_root)/'output'/'assistant_checks'
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='check-', dir=root))
    outcomes = []
    for label, config in (('baseline', baseline), ('candidate', candidate)):
        if cancelled.is_set():
            raise ComparisonCancelled()
        directory = folder/label
        directory.mkdir()
        config_file, result_file = directory/'config.json', directory/'result.json'
        config_file.write_text(json.dumps(config, allow_nan=False))
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                   '--config', str(config_file), '--output', str(result_file), '--mode', proposal['mode']]
        environment={**os.environ, 'MPLBACKEND':'Agg', 'HYBRID_SIM_PROGRESS':'0'}
        environment.pop('HYBRID_STOP_FILE',None)
        with (directory/'calculation.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment,
                                       start_new_session=os.name == 'posix')
            started, last_notice = time.monotonic(), -1
            try:
                while True:
                    if cancelled.is_set():
                        raise ComparisonCancelled()
                    elapsed = time.monotonic()-started
                    if elapsed > timeout:
                        raise RuntimeError('Numerical comparison timed out. No tested recommendation is available.')
                    if on_progress and int(elapsed) != last_notice:
                        last_notice = int(elapsed)
                        on_progress(f'Testing {"current" if label == "baseline" else "proposed"} settings ({last_notice}s)...')
                    try:
                        code = process.wait(timeout=.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if code:
                    raise RuntimeError(f'Numerical comparison failed; see {directory / "calculation.log"}. No values were applied.')
                outcomes.append(json.loads(result_file.read_text()))
            finally:
                _stop(process)
    if cancelled.is_set():
        raise ComparisonCancelled()
    result = compare_results(proposal, *outcomes)
    result['source'] = str((folder/'comparison.json').resolve())
    Path(result['source']).write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def checked_answer(proposal, comparison):
    """Present deterministic measured values; keep Apply disabled on regression."""
    # Recheck provenance and scores rather than trusting an external conclusion.
    verified = compare_results(proposal, comparison['baseline'], comparison['candidate'])
    lines = ['I simulated the current and proposed settings with the same storage, noise and tolerances.', '',
             '| Parameter | Current | Tested |', '|---|---:|---:|']
    for key, value in proposal['changes'].items():
        lines.append(f"| {key} | {proposal['base_settings'][key]} | {value} |")
    lines += ['', '| Direction | Minimum logical fidelity: current → tested | Selected input: current → tested |', '|---|---:|---:|']
    for direction in MODES[proposal['mode']]:
        before, after = [comparison[key]['directions'][direction] for key in ('baseline', 'candidate')]
        lines.append(f"| {direction} | {min(before['sampled_logical_lows']):.4%} → {min(after['sampled_logical_lows']):.4%} | "
                     f"{before['selected_input_fidelity']:.4%} → {after['selected_input_fidelity']:.4%} |")
    lines += ['', f"Comparison score: **{verified['baseline_score']:.4%} → {verified['candidate_score']:.4%}**.",
              verified['objective']]
    if verified['improved']:
        lines += ['', 'The tested proposal improves this score. It is available for preview; the panel is unchanged.']
    else:
        lines += ['', 'The proposal did not improve this score by more than 0.001 percentage points. '
                      'I am keeping your current values; there is no improved proposal to apply.']
    lines += ['', verified['limitation'], f"Numerical evidence: {comparison.get('source', 'local comparison')}"]
    if verified['improved']:
        lines += ['', '```panel-settings', json.dumps(dict(changes=proposal['changes'], evidence='measured simulation result',
                    reason=f"Tested comparison score improved from {verified['baseline_score']:.6%} to {verified['candidate_score']:.6%}; only these two configurations were compared.")), '```']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=tuple(MODES), required=True)
    args = parser.parse_args()
    # backend initializes its operators from --config at import time.
    sys.argv = ['backend.py', '--config', args.config]
    import backend as b
    result = evaluate_backend(b, args.mode, Path(args.output).parent)
    Path(args.output).write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
