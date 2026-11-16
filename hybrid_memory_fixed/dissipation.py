"""Configurable local Markovian channels; rates in 1/us, never rad/us.

Unspecified additions are zero. No atomic data or bath temperature is inferred.
The empty configuration exactly preserves the original collapse operators.
"""
import json
import math


DEFAULT_DISSIPATION = "{}"


def parse_dissipation(value, nt, levels):
    """Validate JSON and references before building any operators."""
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError) as exc:
        raise ValueError("DISSIPATION must be a JSON object.") from exc
    defaults = dict(cavity_nbar=0.0, transmon_nbar=0.0,
                    transmon_down_rates_per_us=[], atomic_branching={},
                    atomic_transitions=[], atomic_projector_dephasing_per_us={},
                    flux_white_psd=0.0, field_white_psd=0.0,
                    optical_scattering=[])
    if not isinstance(raw, dict) or set(raw)-set(defaults):
        raise ValueError("DISSIPATION has unknown keys or is not an object.")
    c = dict(defaults, **raw)

    def rate(value):
        if isinstance(value, bool):
            raise ValueError("Dissipation rates must be finite nonnegative numbers.")
        try:
            number = float(value)
        except (ValueError, TypeError):
            raise ValueError("Dissipation rates must be finite nonnegative numbers.") from None
        if not math.isfinite(number) or number < 0:
            raise ValueError("Dissipation rates must be finite nonnegative numbers.")
        return number

    for key in ('cavity_nbar', 'transmon_nbar', 'flux_white_psd', 'field_white_psd'):
        c[key] = rate(c[key])
    rates = c['transmon_down_rates_per_us']
    if not isinstance(rates, list) or (rates and len(rates) != nt-1):
        raise ValueError("transmon_down_rates_per_us must be empty or have NT-1 rates (1->0, 2->1, ...).")
    c['transmon_down_rates_per_us'] = [rate(x) for x in rates]
    names = {'h0', 'h1', 'r0', 'r1', 'wrong', 'lost'} | {x['label'] for x in levels}
    rydberg = names - {'h0', 'h1', 'lost'}
    branching = c['atomic_branching']
    if not isinstance(branching, dict) or set(branching)-rydberg:
        raise ValueError("atomic_branching sources must be defined Rydberg levels.")
    c['atomic_branching'] = {}
    for source, destinations in branching.items():
        if not isinstance(destinations, dict) or not destinations or set(destinations)-(names-{source}):
            raise ValueError("Atomic branching needs defined, distinct destination states.")
        fractions = {target: rate(x) for target, x in destinations.items()}
        if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Atomic branching fractions must sum to one (include lost explicitly).")
        c['atomic_branching'][source] = fractions
    dephasing = c['atomic_projector_dephasing_per_us']
    if not isinstance(dephasing, dict) or set(dephasing)-(names-{'lost'}):
        raise ValueError("Atomic projector dephasing must reference defined non-loss states.")
    c['atomic_projector_dephasing_per_us'] = {name: rate(x) for name, x in dephasing.items()}
    for key in ('atomic_transitions', 'optical_scattering'):
        if not isinstance(c[key], list):
            raise ValueError(key + " must be a list.")
        result = []
        for entry in c[key]:
            fields = {'source', 'target', 'rate_per_us'} | ({'arm'} if key == 'optical_scattering' else set())
            if not isinstance(entry, dict) or set(entry) != fields:
                raise ValueError(key + " requires exactly " + ', '.join(sorted(fields)))
            source, target = entry['source'], entry['target']
            if not isinstance(source, str) or not isinstance(target, str) or source not in names-{'lost'} or target not in names:
                raise ValueError(key + " must reference defined atomic states; lost is absorbing.")
            if key == 'atomic_transitions' and source == target:
                raise ValueError("Use projector dephasing for elastic atomic channels.")
            item = dict(entry, rate_per_us=rate(entry['rate_per_us']))
            if key == 'optical_scattering' and (isinstance(entry['arm'], bool) or entry['arm'] not in (0, 1)):
                raise ValueError("Optical arm must be 0 or 1.")
            result.append(item)
        c[key] = result
    return c


def collapse_operators(b, flux, field, optical0, optical1):
    """Construct independent jumps using controls for this particular protocol.

    White noise convention: <dx(t) dx(t')> = S_x delta(t-t').
    Thus c_x(t) = sqrt(S_x) d(H/hbar)/dx; no extra 2*pi on rates.
    Optical rates multiply (Omega_arm/Omega_peak)^2.
    """
    import numpy as np
    import qutip as qt

    c = parse_dissipation(b._SETTINGS['DISSIPATION'], b.NT, b.EXTRA_LEVELS)
    ops = []
    rates = c['transmon_down_rates_per_us']
    if rates:
        for j, gamma in enumerate(rates, start=1):
            jump = qt.tensor(qt.basis(b.NT, j-1)*qt.basis(b.NT, j).dag(), b.IC, b.IA)
            ops.append(np.sqrt(gamma*(1+c['transmon_nbar']))*jump)
            if c['transmon_nbar']:
                ops.append(np.sqrt(gamma*c['transmon_nbar'])*jump.dag())
    else:
        ops.append(np.sqrt((1+c['transmon_nbar'])/b.T1_T)*b.sm)
        if c['transmon_nbar']:
            ops.append(np.sqrt(c['transmon_nbar']/b.T1_T)*b.sm.dag())
    ops.append(np.sqrt((1+c['cavity_nbar'])/b.T1_C)*b.a)
    if c['cavity_nbar']:
        ops.append(np.sqrt(c['cavity_nbar']/b.T1_C)*b.a.dag())
    lifetimes = dict(r0=b.TAU_R0, r1=b.TAU_R1, wrong=b.TAU_WRONG,
                     **{x['label']: x['lifetime_us'] for x in b.EXTRA_LEVELS})
    for source, lifetime in lifetimes.items():
        for target, fraction in c['atomic_branching'].get(source, {'lost': 1.0}).items():
            ops.append(np.sqrt(fraction/lifetime)*b.A(b.ATOM_STATES[target], b.ATOM_STATES[source]))
    ops.extend([np.sqrt(1/(2*b.TPHI_T))*b.Zt,
                np.sqrt(1/(2*b.TPHI_R))*b.Zr,
                np.sqrt(1/(2*b.TPHI_H))*b.Zh])
    for entry in c['atomic_transitions']:
        ops.append(np.sqrt(entry['rate_per_us'])*b.A(b.ATOM_STATES[entry['target']], b.ATOM_STATES[entry['source']]))
    for name, gamma in c['atomic_projector_dephasing_per_us'].items():
        ops.append(np.sqrt(gamma)*b.A(b.ATOM_STATES[name], b.ATOM_STATES[name]))
    if c['flux_white_psd']:
        def flux_amplitude(t):
            phi = flux(t)
            u = np.cos(np.pi*phi)**2+b.ASYMMETRY**2*np.sin(np.pi*phi)**2
            slope = (np.sqrt(8*b.EC*b.EJ_MAX)*.25*u**(-.75)
                     *np.pi*(b.ASYMMETRY**2-1)*np.sin(2*np.pi*phi))
            return np.sqrt(c['flux_white_psd'])*b.TWOPI*slope
        ops.append(qt.QobjEvo([b.Nt, flux_amplitude]))
    if c['field_white_psd']:
        def field_operator(t):
            return np.sqrt(c['field_white_psd'])*b.atomic_field_derivative(t, field)
        ops.append(qt.QobjEvo(field_operator))
    for entry in c['optical_scattering']:
        jump = b.A(b.ATOM_STATES[entry['target']], b.ATOM_STATES[entry['source']])
        drive = (optical0, optical1)[entry['arm']]
        def amplitude(t, gamma=entry['rate_per_us'], envelope=drive):
            return np.sqrt(gamma)*abs(envelope(t))/b.OMEGA
        ops.append(qt.QobjEvo([jump, amplitude]))
    return ops


def snapshot(operators, time):
    """Freeze time-dependent channels at parked controls for constant propagation."""
    import qutip as qt
    return [op(time) if isinstance(op, qt.QobjEvo) else op for op in operators]
