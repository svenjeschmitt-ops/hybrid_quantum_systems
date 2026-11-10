"""Sequential parameter sweeps; each point loads a fresh backend model."""
import csv
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime
import uuid

from hybrid_settings import FIELDS, validate, derived

SWEEP_FIELDS = [row for row in FIELDS
                if row[0] not in ('Experimental tolerances', 'Simultaneous resonance', 'Numerics')
                and row[-1] != 'boolean']
MAX_POINTS = 501


def sweep_configs(config, parameter, start, stop, step):
    """Return validated points on start + n*step, without overshooting stop."""
    if parameter not in {row[1] for row in SWEEP_FIELDS}:
        raise ValueError('Choose a supported continuous model parameter.')
    try:
        start, stop, step = map(float, (start, stop, step))
    except (TypeError, ValueError):
        raise ValueError('Sweep start, stop and step must be numbers.') from None
    if not all(math.isfinite(x) for x in (start, stop, step)) or step == 0:
        raise ValueError('Sweep values must be finite and step must be nonzero.')
    if start == stop or (stop - start) / step <= 0:
        raise ValueError('Use distinct start/stop values and a step pointing toward stop.')
    intervals = (stop - start) / step
    if not math.isfinite(intervals) or intervals > MAX_POINTS - 1 + 1e-9:
        raise ValueError(f'Use at most {MAX_POINTS} sweep points; increase the step.')
    count = math.floor(intervals + 1e-10) + 1
    if count < 2:
        raise ValueError('The step must produce at least two sweep points.')
    base = validate(config)
    if parameter in ('G_TC_MHZ', 'G_AC_MHZ') and not base['DIRECT_COUPLING']:
        raise ValueError('Enable Direct information for coupling before sweeping manual coupling strengths.')
    if parameter in ('BETA', 'V_ZPF_UV', 'D_01_EA0', 'E_ZPF_V_PER_M', 'DIPOLE_ALIGNMENT') and base['DIRECT_COUPLING']:
        raise ValueError('Turn off Direct information for coupling to sweep physical coupling inputs.')
    points = []
    for index in range(count):
        value = start + index * step
        if math.isclose(value, stop, rel_tol=1e-12, abs_tol=1e-14):
            value = stop
        try:
            points.append(validate({**base, parameter: value}))
        except ValueError as exc:
            raise ValueError(f'Sweep point {index + 1} ({parameter}={value:g}): {exc}') from exc
    return points


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def evaluate_point(b, directions):
    """Selected-input dynamics and unconditional joint overlaps, with storage."""
    import matplotlib.pyplot as plt
    c = b._SETTINGS
    previous_output = b._ASSISTANT_OUTPUT
    result = {'settings': c, 'derived': derived(c), 'directions': {}}
    # Reuse the ordinary figures' calculations and reports so tables, basis
    # outputs and dynamics have exactly the same definitions as single runs.
    # Each point retains its configured bounds; error scenarios use that
    # point's fixed nominal phase, couplings and pulse timing.
    try:
        with tempfile.TemporaryDirectory(prefix='hybrid-sweep-report-') as directory:
            b._ASSISTANT_OUTPUT = directory
            for direction in directions:
                previous_figures = set(plt.get_fignums())
                try:
                    if direction == 'write':
                        b.main(save=False)
                    else:
                        b.retrieval_main(save=False)
                    payload = json.loads((Path(directory)/(direction+'.json')).read_text())
                finally:
                    for number in set(plt.get_fignums()) - previous_figures:
                        plt.close(number)
                payload['settings'] = c
                dynamics = payload['traces']['selected_input_dynamics']
                series = dynamics['series']
                summary = payload['summary']
                final = (summary['selected_fidelity_after_storage'] if direction == 'write'
                         else summary['selected_joint_fidelity'])
                names = {'transmon_e': 'transmon_e', 'cavity_photons': 'cavity_mean_photons',
                         'atom_h0': 'atom_h0', 'atom_h1': 'atom_h1', 'atom_r0': 'atom_r0',
                         'atom_r1': 'atom_r1', 'atom_loss': 'atom_loss',
                         'atom_wrong': 'atom_wrong_rydberg'}
                result['directions'][direction] = {
                    'final_fidelity': final,
                    'fidelity_range': (summary['tolerance_ranges']['metric_ranges']['final_fidelity']
                                       if summary.get('tolerance_ranges') else {'low': final, 'high': final}),
                    'phase_correction_rad': payload['metadata']['phase_correction_rad'],
                    'time_us': dynamics['time_us'], 'joint_fidelity': series['joint_fidelity'],
                    'populations': {name: series[key] for name, key in names.items()},
                    'figure_report': payload,
                }
    finally:
        b._SETTINGS = c
        b._ASSISTANT_OUTPUT = previous_output
    return result


def run(backend, config, parameter, start, stop, step, direction='both',
        save=False, output_directory=None):
    if direction not in ('both', 'write', 'read'):
        raise ValueError('Sweep direction must be both, write or read.')
    points = sweep_configs(config, parameter, start, stop, step)
    directions = ('write', 'read') if direction == 'both' else (direction,)
    folder = Path(output_directory) if output_directory else (
        Path(backend.__file__).resolve().parent.parent / 'output' / 'sweep_runs' /
        (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]))
    folder.mkdir(parents=True, exist_ok=True)
    unit = next(row[3] for row in SWEEP_FIELDS if row[1] == parameter)
    from tolerance_scan import AXES
    tolerance_count = 3 ** sum(points[0][key] > 0 for _, key, _ in AXES)
    report = {'settings': validate(config), 'parameter': parameter, 'unit': unit,
              'requested_range': {'start': float(start), 'stop': float(stop), 'step': float(step)},
              'total_points': len(points), 'status': 'running', 'results': [],
              'metric': 'Selected-input unconditional joint fidelity, with configured noise. '
                        'Write includes subsequent storage; independent read input includes prior storage.',
              'calibration': 'Fresh model per point. Derived couplings and pulse timings are recomputed. '
                             'A noiseless phase reference is calibrated separately at each point. '
                             "Within each point, tolerance scenarios keep that point's nominal calibration fixed. No pulse optimization.",
              'trace_times': 'Write traces end before storage; final write fidelity is after storage. '
                             'Read trace time starts after storage.',
              'tolerances': 'Configured fixed +/- bounds are scanned around every sweep point. '
                            'Shading and summary ranges show sampled extrema, not guaranteed bounds.',
              'tolerance_points_per_sweep_point': tolerance_count}
    summary_path = folder / 'parameter_sweep.json'
    atomic_json(summary_path, report)
    print(f'Parameter sweep: {parameter}, {len(points)} points; {direction}.', flush=True)
    print(report['calibration'], flush=True)
    print(f'Tolerance grid: {tolerance_count} fixed-error scenarios per sweep point per direction '
          '(including nominal).', flush=True)
    csv_path = folder / 'parameter_sweep.csv'
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[parameter] + [d+suffix for d in directions
                for suffix in ('_fidelity', '_fidelity_low', '_fidelity_high')])
            writer.writeheader()
            csv_file.flush()
            for index, point in enumerate(points):
                value = point[parameter]
                print(f'Sweep point {index+1}/{len(points)}: {parameter}={value:.12g}', flush=True)
                point_path = folder / f'point_{index:04d}.json'
                with tempfile.TemporaryDirectory(prefix='hybrid-sweep-') as temporary:
                    config_path = Path(temporary) / 'settings.json'
                    config_path.write_text(json.dumps(point), encoding='utf-8')
                    command = [sys.executable, '-u', str(Path(backend.__file__).resolve()),
                               '--run', '--config', str(config_path), '--sweep-point', str(point_path),
                               '--sweep-direction', direction, '--no-show']
                    env = {**os.environ, 'MPLBACKEND': 'Agg'}
                    subprocess.run(command, check=True, env=env)
                data = json.loads(point_path.read_text(encoding='utf-8'))
                row = {parameter: value, **{d+'_fidelity': data['directions'][d]['final_fidelity'] for d in directions}}
                for d in directions:
                    bounds = data['directions'][d].get('fidelity_range',
                              {'low': row[d+'_fidelity'], 'high': row[d+'_fidelity']})
                    row.update({d+'_fidelity_low': bounds['low'], d+'_fidelity_high': bounds['high']})
                writer.writerow(row)
                csv_file.flush()
                report['results'].append({**row, 'data_file': point_path.name})
                atomic_json(summary_path, report)
                print('  '+', '.join(f'{d}={100*row[d+"_fidelity"]:.6f}%' for d in directions), flush=True)
    except BaseException:
        report['status'] = 'incomplete'
        atomic_json(summary_path, report)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    report['status'] = 'complete'
    atomic_json(summary_path, report)
    from sweep_figure import SweepFigure
    for d in directions:
        viewer = SweepFigure(backend, report, folder, d)
        if save:
            viewer.figure.savefig(folder / f'parameter_sweep_{d}.png', dpi=170)
    print(f'Full parameter sweep results: {summary_path}', flush=True)
    return report
