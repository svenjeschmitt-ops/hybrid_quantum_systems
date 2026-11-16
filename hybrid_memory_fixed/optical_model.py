"""A shared intermediate-state optical Hamiltonian and its effective reduction.

Convention: H_p/hbar = (Delta + carrier_reference) P_p and
V_plus = (Omega_lower exp(-i carrier*t)|p><h| + Omega_upper|p><R|)/2.
Delta is signed; linewidth is total population decay in 1/us.
"""
import json
import math
import numpy as np
from multilevel import parse_manifold
from atomic_model import AtomicModel


def parse_optics(value, config):
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError) as exc:
        raise ValueError('OPTICAL_SCHEME must be a JSON object.') from exc
    if not isinstance(raw, dict):
        raise ValueError('OPTICAL_SCHEME must be a JSON object.')
    if not raw:
        return {}
    if set(raw) != {'mode', 'provenance', 'arms'} or raw['mode'] not in ('effective', 'explicit'):
        raise ValueError('OPTICAL_SCHEME requires mode (effective/explicit), provenance and arms.')
    if not isinstance(raw['provenance'], str) or not raw['provenance'].strip():
        raise ValueError('OPTICAL_SCHEME requires a provenance description.')
    if not isinstance(raw['arms'], dict) or set(raw['arms']) != {'0', '1'}:
        raise ValueError('OPTICAL_SCHEME requires both optical arms, named "0" and "1".')
    names = {'h0', 'h1', 'r0', 'r1', 'wrong', 'lost'} | {x['label'] for x in parse_manifold(config['RYDBERG_MANIFOLD'])}
    if {'optical_p0', 'optical_p1'} & names:
        raise ValueError('optical_p0 and optical_p1 are reserved intermediate-state labels.')
    result = dict(mode=raw['mode'], provenance=raw['provenance'], arms={})
    for arm, spec in raw['arms'].items():
        required = {'detuning_MHz', 'lower_rabi_MHz', 'upper_rabi_MHz', 'linewidth_per_us', 'branching'}
        if not isinstance(spec, dict) or set(spec) != required:
            raise ValueError('Each optical arm requires exactly: '+', '.join(sorted(required)))
        item = {}
        for key in required-{'branching'}:
            try:
                number = float(spec[key])
            except (TypeError, ValueError):
                raise ValueError('Optical parameters must be finite numbers.') from None
            if not math.isfinite(number):
                raise ValueError('Optical parameters must be finite numbers.')
            item[key] = number
        if item['detuning_MHz'] == 0 or item['lower_rabi_MHz'] == 0 or item['upper_rabi_MHz'] == 0 or item['linewidth_per_us'] < 0:
            raise ValueError('Optical detuning and Rabi frequencies must be nonzero; linewidth must be nonnegative.')
        fractions = spec['branching']
        if not isinstance(fractions, dict) or not fractions or set(fractions)-names:
            raise ValueError('Intermediate decay branching must reference defined atomic states.')
        try:
            fractions = {name: float(fraction) for name, fraction in fractions.items()}
        except (TypeError, ValueError):
            raise ValueError('Optical branching fractions must be finite nonnegative numbers.') from None
        if any(not math.isfinite(x) or x < 0 for x in fractions.values()) or not math.isclose(sum(fractions.values()), 1., abs_tol=1e-12, rel_tol=0):
            raise ValueError('Optical branching fractions must be nonnegative and sum to one.')
        item['branching'] = fractions
        if raw['mode'] == 'effective':
            relatives = [1., config['REL_WRONG_DIPOLE']] if arm == '0' else [1.]
            relatives += [x['optical_relative'] for x in parse_manifold(config['RYDBERG_MANIFOLD']) if str(x['branch']) == arm]
            upper_norm = abs(item['upper_rabi_MHz'])*np.linalg.norm(relatives)
            scale = max(abs(item['lower_rabi_MHz']), upper_norm, item['linewidth_per_us']/(2*np.pi))
            if abs(item['detuning_MHz']) < 10*scale:
                raise ValueError('Effective optics requires |detuning| >= 10 times the Rabi/linewidth scale; use explicit mode otherwise.')
        result['arms'][arm] = item
    peaks = [effective_peak_mhz(result['arms'][arm]) for arm in ('0', '1')]
    if not math.isclose(peaks[0], peaks[1], rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError('Both optical arms must have equal effective peak Rabi magnitudes for the shared pulse duration.')
    if peaks[0] <= 0 or not math.isfinite(peaks[0]):
        raise ValueError('Derived optical Rabi frequency must be finite and positive.')
    if result['mode'] == 'effective':
        atom = AtomicModel(config)
        swap = config['V_SWAP']/config['GAP_CM']+config['STRAY_FIELD']
        park = swap if config['CONSTANT_FIELD'] else config['E_PARK']
        # Optical pulses occur at nominal park (or at the held constant field),
        # before/after exchange excursions. Include static field uncertainty.
        fields = np.linspace(park-config['EFIELD_TOL'], park+config['EFIELD_TOL'], 3)
        levels = parse_manifold(config['RYDBERG_MANIFOLD'])
        for index in (0, 1):
            participating = [f'r{index}']
            if index == 0 and config['REL_WRONG_DIPOLE']:
                participating.append('wrong')
            participating += [x['label'] for x in levels if x['branch'] == index and x['optical_relative']]
            reference = atom.energy(f'r{index}', park)
            spread = max(float(np.max(abs(atom.energy(name, fields)-reference))) for name in participating)
            if abs(result['arms'][str(index)]['detuning_MHz']) < 10*max(spread, peaks[index]):
                raise ValueError('Effective optical detuning must exceed retained atomic energy offsets and pulse bandwidth by a factor of 10; use explicit mode otherwise.')
    return result


def effective_peak_mhz(arm):
    """Magnitude of twice the eliminated off-diagonal element, in ordinary MHz."""
    delta = 2*np.pi*arm['detuning_MHz']
    gamma = arm['linewidth_per_us']
    return abs(delta*(2*np.pi*arm['lower_rabi_MHz'])*(2*np.pi*arm['upper_rabi_MHz'])
               /(2*(delta**2+(gamma/2)**2)))/(2*np.pi)


def optical_settings(config):
    value = config.get('OPTICAL_SCHEME', '{}')
    return json.loads(value) if isinstance(value, str) else value


def optical_peak(config):
    spec = optical_settings(config)
    return effective_peak_mhz(spec['arms']['0']) if spec else config['OMEGA']


def build_optics(b, pulses, nominal_field):
    """Return H/hbar and collapse operators for a protocol's optical intervals.

    A pulse is (arm index, (start, end)). Both propagation directions use the
    same Hermitian optical Hamiltonian. Carrier phase origins follow each pulse.
    """
    import qutip as qt
    spec = optical_settings(b._SETTINGS)
    zero = 0*b.Pr0
    h = qt.QobjEvo(zero)
    collapse = []
    for arm in (0, 1):
        intervals = [interval for index, interval in pulses if index == arm]
        ground = b.ATOM_STATES[f'h{arm}']
        carrier = b.TWOPI*b.atomic_energy_MHz(f'r{arm}', nominal_field, offsets=False)
        def normalized(t, interval):
            value = b.optical_envelope(t, *interval)/b.OMEGA
            # Each microscopic Rabi follows sin, so their product follows the
            # original sin^2 effective pulse and retains its pi-area duration.
            return np.sqrt(value) if spec else value
        def shape(t, windows=intervals, envelope=normalized):
            return sum(envelope(t, interval) for interval in windows)
        def phased_shape(t, windows=intervals, reference=carrier, envelope=normalized):
            return sum(envelope(t, interval)*np.exp(-1j*reference*(t-interval[0]))
                       for interval in windows)
        transition = b.Bprep if arm == 0 else b.Bmap1.dag()
        if not spec:
            coefficient = lambda t, f=phased_shape: .5*b.OMEGA*f(t)
            drive = qt.QobjEvo([transition, coefficient])
            h += drive+drive.dag()
            continue
        parameters = spec['arms'][str(arm)]
        delta, gamma = b.TWOPI*parameters['detuning_MHz'], parameters['linewidth_per_us']
        lower, upper = b.TWOPI*parameters['lower_rabi_MHz'], b.TWOPI*parameters['upper_rabi_MHz']
        # Use the ground ket as a dummy intermediate row in the reduced model.
        intermediate = b.ATOM_STATES[f'optical_p{arm}'] if spec['mode'] == 'explicit' else ground
        # transition has support on one atomic source; recover its coherent Rydberg vector.
        rvector = b.r0 if arm == 0 else b.r1
        if arm == 0:
            rvector = rvector+b.REL_WRONG_DIPOLE*b.wrong
        for level in b.EXTRA_LEVELS:
            if level['branch'] == arm:
                rvector += level['optical_relative']*b.ATOM_STATES[level['label']]
        v = qt.QobjEvo([[b.A(intermediate, ground), lambda t, f=phased_shape, omega=lower: .5*omega*f(t)],
                       [b.A(intermediate, rvector), lambda t, f=shape, omega=upper: .5*omega*f(t)]])
        if spec['mode'] == 'explicit':
            h += (delta+carrier)*b.A(intermediate, intermediate)+v+v.dag()
            for target, fraction in parameters['branching'].items():
                collapse.append(np.sqrt(gamma*fraction)*b.A(b.ATOM_STATES[target], intermediate))
        else:
            denominator = delta**2+(gamma/2)**2
            h += (-delta/denominator)*(v.dag()*v)
            for target, fraction in parameters['branching'].items():
                # A single coherent jump per physical decay destination, including
                # both virtual excitation paths and their relative optical phase.
                collapse.append(np.sqrt(gamma*fraction)/(delta-1j*gamma/2)
                                *b.A(b.ATOM_STATES[target], intermediate)*v)
    return h, collapse
