"""Nominal pulse controls for the interactive timing preview."""

import numpy as np

from atomic_model import AtomicModel
from pulse_shapes import parse_shapes, pulse_window
from hybrid_settings import (calculated_swap_field, calculated_transmon_frequency,
                             derived, effective_controls, pulse_intervals, validate,
                             constant_field_intervals)


def control_preview(settings, direction='write', times=None, report=None):
    """Show nominal controls or the exact controls from a matching run report."""
    config = validate(settings)
    if direction not in ('write', 'read'):
        raise ValueError('Preview direction must be write or read')
    controls = effective_controls(config)
    values = derived(config)
    optical0 = values['optical_pi_pulse_us']
    optical1 = values['optical_pi_pulse_1_us']
    prep = (0., optical0 if direction == 'write' else max(optical0, optical1))
    tc, ac = pulse_intervals(config, prep[1]+config['GAP'],
                            values['transmon_pulse_duration_us'],
                            values['atom_pulse_duration_us'], direction)
    mapping_start = max(tc[1], ac[1])+config['GAP']
    if config['CONSTANT_FIELD']:
        tc, ac, mapping_start = constant_field_intervals(
            config, prep[1]+config['GAP'] if config['CONSTANT_FLUX'] or direction == 'read' else tc[0],
            values['transmon_pulse_duration_us'], direction)
    mapping = (mapping_start, mapping_start+(max(optical0, optical1) if direction == 'write' else optical0))
    if report is not None:
        if (report.get('schema_version') != 1 or report.get('direction') != direction or
                validate(report.get('settings')) != config):
            raise ValueError('Run report does not match the preview settings and direction')
        actual = report['metadata']['stages_us']
        if direction == 'write':
            prep = tuple(actual['preparation'])
            tc, ac, mapping = (tuple(actual[key]) for key in ('transmon_cavity','atom_cavity','mapping'))
        else:
            prep = tuple(actual.get('Memory to Rydberg', (0., 0.)))
            tc, ac, mapping = (tuple(actual[key]) for key in ('Cavity to transmon','Atom to cavity','Return atom to h0'))
        mapping_start = mapping[0]
        times = np.asarray(report['traces']['controls']['time_us'], dtype=float)
    manual_switch = (config['COUPLING_SWITCH_MODE'] == 'manual' and
                     (config['CONSTANT_FLUX'] or config['CONSTANT_FIELD']))
    nominal_switch = (config[direction.upper()+'_COUPLING_START_NS']/1000 if manual_switch else None)
    finish = (max(mapping[1], nominal_switch * 1.025) if report is None and nominal_switch is not None
              else mapping[1])
    if times is None:
        points = [np.linspace(0, finish, 601),
                  np.array([*prep, *tc, *ac, *mapping, sum(tc)/2, sum(ac)/2])]
        intervals = [tc, ac, (0., optical0), (mapping_start, mapping_start+optical0)]
        intervals.append((mapping_start, mapping_start+optical1) if direction == 'write' else (0., optical1))
        points.extend(np.linspace(*interval, 257) for interval in intervals)
        times = np.unique(np.concatenate(points))
        times = times[times <= finish]
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
    if report is not None:
        series = report['traces']['controls']['series']
        flux = np.asarray(series['flux_Phi_over_Phi0'], dtype=float)
        voltage = np.asarray(series['electrode_voltage_V'], dtype=float)
        transmon = np.asarray(series['transmon_detuning_MHz'], dtype=float)
        atom = np.asarray(series['atom_detuning_MHz'], dtype=float)
        if any(values.shape != times.shape for values in (flux,voltage,transmon,atom)):
            raise ValueError('Run report control traces do not match their time axis')
    switch = report.get('summary',{}).get('idealized_coupling_switch') if report else None
    return dict(settings=config, direction=direction, source='simulation' if report is not None else 'nominal',
                coupling_switch_us=(float(switch['time_us']) if switch else nominal_switch if report is None else None),
                time_us=times.tolist(), flux=flux.tolist(), voltage=voltage.tolist(),
                optical0=optical_0.tolist(), optical1=optical_1.tolist(),
                transmon_detuning=transmon.tolist(), atom_detuning=atom.tolist(),
                stages=dict(preparation=prep, flux=tc, voltage=ac, mapping=mapping),
                delay_ns=1000*((ac[0]+ac[1])-(tc[0]+tc[1]))/2,
                overlap_ns=1000*max(0., min(tc[1], ac[1])-max(tc[0], ac[0])),
                finish_us=finish)
