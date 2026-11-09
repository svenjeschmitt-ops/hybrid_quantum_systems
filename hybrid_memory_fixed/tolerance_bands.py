"""Pointwise static-error envelopes for the ordinary transfer figures."""
import numpy as np

from simulation_output import trace
from tolerance_scan import AXES, fixed_offsets, scenarios

POPULATION_NAMES = ('transmon_e', 'cavity_mean_photons', 'atom_h0', 'atom_h1',
                    'atom_r0', 'atom_r1', 'atom_loss', 'atom_wrong_rydberg')


def enabled(config):
    return any(config[key] > 0 for _, key, _ in AXES)


def controls(b, times, mode, source):
    if mode == 'write':
        flux, voltage = b.flux, b.electrode_voltage
    else:
        _, _, _, flux, voltage = b.retrieval_model(source)
    phi = np.array([flux(t) for t in times])
    field = np.array([voltage(t) / b.GAP_CM + b.STRAY_FIELD
                      + b.TOLERANCE_OFFSETS['EFIELD'] for t in times])
    dc = b.TOLERANCE_OFFSETS['FC']
    return dict(flux=phi, field=field,
                transmon_detuning=b.transmon_frequency(phi) + b.TOLERANCE_OFFSETS['FT'] - b.FC - dc,
                atom_detuning=b.F_A_ZERO + b.TOLERANCE_OFFSETS['FA'] - b.FC - dc
                + (b.K_R1 - b.K_R0) * field**2)


def final_metrics(b, state, mode, fidelity, before_fidelity, basis):
    """Evaluate combined quantities in each scenario before taking extrema."""
    subsystem = state.ptrace(2 if mode == 'write' else 0)
    metrics = dict(final_fidelity=float(fidelity), before_fidelity=float(before_fidelity),
                   coherence=float(2 * abs(subsystem[0, 1])),
                   rydberg_population=float(np.real(b.qt.expect(b.Pr0 + b.Pr1, state))))
    for name, op in zip(POPULATION_NAMES, (b.Pe, b.n, b.Ph0, b.Ph1,
                                          b.Pr0, b.Pr1, b.Plost, b.Pwrong)):
        metrics[name] = float(np.real(b.qt.expect(op, state)))
    metrics.update(zip(('basis_0', 'basis_1', 'basis_plus'), map(float, basis)))
    return metrics


def compute(b, config, mode, alpha, beta, phase, times, nominal_pops,
            nominal_fidelity, nominal_final, source='hyperfine',
            nominal_state=None, nominal_basis=None):
    """Reuse nominal data; evolve selected and logical inputs at each error point.

    Min/max at each time may come from different offset combinations. These
    envelopes are not individual realizable trajectories or confidence bands.
    """
    if not enabled(config):
        return None
    if nominal_state is None or nominal_basis is None:
        raise ValueError('Tolerance summaries require the nominal final state and basis fidelities.')
    target = (alpha * b.g + beta * b.e).unit()
    points = list(scenarios(config))
    values = dict(populations=np.asarray(nominal_pops), fidelity=np.asarray(nominal_fidelity),
                  **controls(b, times, mode, source))
    low = {key: np.array(value, copy=True) for key, value in values.items()}
    high = {key: np.array(value, copy=True) for key, value in values.items()}
    endpoints = [dict(offsets=points[0], final_fidelity=float(nominal_final),
                      metrics=final_metrics(b, nominal_state, mode, nominal_final,
                                            nominal_fidelity[-1], nominal_basis))]
    print(f'{mode}: calculating tolerance bands for {len(points)} fixed-error points. '
          'Lines are nominal; shaded bands are sampled pointwise ranges.', flush=True)
    ops = (b.Pe, b.n, b.Ph0, b.Ph1, b.Pr0, b.Pr1, b.Plost, b.Pwrong)
    for index, offsets in enumerate(points[1:], start=2):
        print(f'{mode} tolerance point {index}/{len(points)}: {offsets}', flush=True)
        with fixed_offsets(b, offsets):
            if mode == 'write':
                grid, states = b.simulate(alpha, beta, noisy=True, include_storage=False)
                effect = b.atomic_storage_effect(target, phase)
                fidelity = np.real(b.qt.expect(effect, states))
                final_state = b.hold_memory(states[-1])
                final = float(np.real(b.qt.expect(effect, final_state)))
            else:
                initial = b.stored_retrieval_initial(alpha, beta, source)
                grid, states, _, _, _ = b.evolve_read(initial, source=source)
                fidelity = np.asarray(b.read_metrics(states, target, phase)[2])
                final_state = states[-1]
                final = float(fidelity[-1])
            if not np.array_equal(grid, times):
                raise ValueError('Tolerance trajectories must share the nominal time grid.')
            _, _, basis = b.basis_probabilities(mode, source=source, phase=phase, storage=(mode == 'write'))
            metrics = final_metrics(b, final_state, mode, final, fidelity[-1], basis)
            sample = dict(populations=np.real([b.qt.expect(op, states) for op in ops]),
                          fidelity=fidelity, **controls(b, times, mode, source))
        for key, value in sample.items():
            low[key] = np.minimum(low[key], value)
            high[key] = np.maximum(high[key], value)
        endpoints.append(dict(offsets=offsets, final_fidelity=final, metrics=metrics))
    worst = min(endpoints, key=lambda point: point['final_fidelity'])
    best = max(endpoints, key=lambda point: point['final_fidelity'])
    print(f"{mode} selected-input final sampled fidelity range: "
          f"{worst['final_fidelity']:.4%} to {best['final_fidelity']:.4%}", flush=True)
    metric_ranges = {key: dict(low=min(point['metrics'][key] for point in endpoints),
                               high=max(point['metrics'][key] for point in endpoints))
                     for key in endpoints[0]['metrics']}
    return dict(times=np.asarray(times), low=low, high=high, nominal=values,
                summary=dict(point_count=len(points), best=best, worst=worst, scenarios=endpoints,
                             metric_ranges=metric_ranges,
                             meaning='Selected-input sampled pointwise min/max; not guaranteed continuous bounds. '
                             'Different times can have extrema at different offsets. Fixed nominal calibration.'))


def table_rows(rows, bands, mode):
    """Keep the existing rows; add a dedicated range column to every metric."""
    spec = [('basis_0', '.2%'), ('basis_1', '.2%'), ('basis_plus', '.2%')]
    if mode == 'write':
        spec += [('before_fidelity', '.2%'), ('final_fidelity', '.2%'),
                 ('coherence', '.6f'), ('transmon_e', '.3e'),
                 ('cavity_mean_photons', '.3e'), ('rydberg_population', '.3e'),
                 ('atom_loss', '.3e'), ('atom_wrong_rydberg', '.3e')]
    else:
        spec += [('final_fidelity', '.2%'), ('coherence', '.6f'),
                 ('cavity_mean_photons', '.3e'), ('atom_h0', '.6f'),
                 ('rydberg_population', '.3e'), ('atom_loss', '.3e'),
                 ('atom_wrong_rydberg', '.3e')]
    if len(rows) != 5 + len(spec):
        raise ValueError('Summary rows do not match the metric definitions.')
    output = []
    for index, (label, value) in enumerate(rows):
        if index < 5:
            interval = '—'  # Timing is fixed; no timing uncertainty is modeled.
        elif bands is None:
            interval = f'{value}–{value}'
        else:
            key, formatting = spec[index - 5]
            bounds = bands['summary']['metric_ranges'][key]
            interval = f"{bounds['low']:{formatting}}–{bounds['high']:{formatting}}"
        output.append([label, value, interval])
    return output


def draw(bands, flux_axis, detuning_axis, population_axis, read=False):
    if bands is None:
        return
    ts, low, high = bands['times'], bands['low'], bands['high']
    flux_axis.fill_between(ts, low['flux'], high['flux'], color='tab:blue', alpha=.2)
    for key, color in (('transmon_detuning', 'tab:blue'), ('atom_detuning', 'tab:orange')):
        detuning_axis.fill_between(ts, low[key], high[key], color=color, alpha=.2)
    # Read figure uses a different order for its nominal population lines.
    order = (0, 1, 4, 5, 2, 3, 7, 6) if read else tuple(range(8))
    for line, index in zip(population_axis.lines[:8], order):
        population_axis.fill_between(ts, low['populations'][index], high['populations'][index],
                                     color=line.get_color(), alpha=.18)
    population_axis.set_title(population_axis.get_title() + ' — nominal lines, sampled tolerance bands')


def report_traces(bands):
    if bands is None:
        return {}
    series = {}
    for side in ('low', 'high'):
        for key, values in bands[side].items():
            if key == 'populations':
                series.update({name + '_' + side: value
                               for name, value in zip(POPULATION_NAMES, values)})
            else:
                series[key + '_' + side] = values
    return {'selected_input_tolerance_ranges': trace(bands['times'], **series)}
