"""Nominal pulse controls for the interactive timing preview."""

import numpy as np

from atomic_model import AtomicModel
from pulse_shapes import parse_shapes, pulse_window
from hybrid_settings import (calculated_swap_field, calculated_transmon_frequency,
                             derived, effective_controls, pulse_intervals, validate)


def control_preview(settings, direction='write', times=None):
    """Use the production pulse timing, Stark spectrum and control definitions."""
    config = validate(settings)
    controls = effective_controls(config)
    values = derived(config)
    optical0 = values['optical_pi_pulse_us']
    optical1 = values['optical_pi_pulse_1_us']
    prep = (0., optical0 if direction == 'write' else max(optical0, optical1))
    tc, ac = pulse_intervals(config, prep[1]+config['GAP'],
                            values['transmon_pulse_duration_us'],
                            values['atom_pulse_duration_us'], direction)
    mapping_start = max(tc[1], ac[1])+config['GAP']
    mapping = (mapping_start, mapping_start+(max(optical0, optical1) if direction == 'write' else optical0))
    finish = mapping[1]
    if times is None:
        points = [np.linspace(0, mapping[1], 601),
                  np.array([*prep, *tc, *ac, *mapping, sum(tc)/2, sum(ac)/2])]
        intervals = [tc, ac, (0., optical0), (mapping_start, mapping_start+optical0)]
        intervals.append((mapping_start, mapping_start+optical1) if direction == 'write' else (0., optical1))
        points.extend(np.linspace(*interval, 257) for interval in intervals)
        times = np.unique(np.concatenate(points))
    times = np.asarray(times, dtype=float)

    shapes = parse_shapes(config['PULSE_SHAPES'])
    def window(interval, channel):
        return pulse_window(times, *interval, shapes, channel)
    flux = controls['PHI_PARK']+(controls['PHI_SWAP']-controls['PHI_PARK'])*window(tc, 'flux')
    field = controls['E_PARK']+(calculated_swap_field(controls)-controls['E_PARK'])*window(ac, 'voltage')
    start = mapping[0]
    if direction == 'write':
        optical_0 = config['OMEGA']*(window((0., optical0), 'optical0') + window((start, start+optical0), 'optical0'))
        optical_1 = config['OMEGA_1']*window((start, start+optical1), 'optical1')
    else:
        optical_0 = config['OMEGA']*(window((0., optical0), 'optical0') + window((start, start+optical0), 'optical0'))
        optical_1 = config['OMEGA_1']*window((0., optical1), 'optical1')
    voltage = config['GAP_CM']*(field-config['STRAY_FIELD'])
    transmon = np.array([calculated_transmon_frequency(config, phi)-config['FC'] for phi in flux])
    atom = np.asarray(AtomicModel(config).transition(field))-config['FC']
    return dict(settings=config, direction=direction,
                time_us=times.tolist(), flux=flux.tolist(), voltage=voltage.tolist(),
                optical0=optical_0.tolist(), optical1=optical_1.tolist(),
                transmon_detuning=transmon.tolist(), atom_detuning=atom.tolist(),
                stages=dict(preparation=prep, flux=tc, voltage=ac, mapping=mapping),
                delay_ns=values[direction+'_pulse_delay_ns'],
                overlap_ns=1000*max(0., min(tc[1], ac[1])-max(tc[0], ac[0])),
                finish_us=finish)
