"""Field-dependent atomic data in a phase-consistent, adiabatic branch basis.

Energies are ordinary MHz relative to the zero-field r0 reference, before
subtracting the cavity rotating frame. Dipoles are real cavity-polarization
projections in e*a0. Tables must not be extrapolated.
"""
import json
import math
from functools import lru_cache
import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly
from multilevel import parse_manifold

DIPOLE_TO_MHZ = 1.602176634e-19 * 5.29177210544e-11 / 6.62607015e-34 / 1e6


def atomic_levels(config):
    return {'r0': (0, 0., config['K_R0']),
            'r1': (1, config['F_A_ZERO'], config['K_R1']),
            'wrong': (0, config['F_WRONG_SPLIT'], config['K_WRONG']),
            **{x['label']: (x['branch'], x['offset_MHz'] + x['branch']*config['F_A_ZERO'],
                             x['stark_MHz_per_Vcm2']) for x in parse_manifold(config['RYDBERG_MANIFOLD'])}}


def cavity_links(config):
    return {'r1<-r0': 1., **{f"{x['label']}<-{source}": strength
            for x in parse_manifold(config['RYDBERG_MANIFOLD'])
            for source, strength in x['cavity_from'].items()}}


def parse_spectrum(value, config):
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError) as exc:
        raise ValueError('ATOMIC_SPECTRUM must be a JSON object.') from exc
    if not isinstance(raw, dict) or set(raw)-{'provenance', 'field_V_per_cm', 'energies_MHz', 'dipoles_ea0'}:
        raise ValueError('ATOMIC_SPECTRUM contains unknown fields.')
    if not raw:
        return {}
    if not isinstance(raw.get('provenance'), str) or not raw['provenance'].strip():
        raise ValueError('ATOMIC_SPECTRUM requires a provenance description.')
    try:
        fields = np.asarray(raw['field_V_per_cm'], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('ATOMIC_SPECTRUM requires an ordered field_V_per_cm grid.') from exc
    if fields.ndim != 1 or len(fields) < 2 or not np.isfinite(fields).all() or np.any(np.diff(fields) <= 0):
        raise ValueError('Atomic field grid must contain at least two finite, strictly increasing values.')
    result = dict(provenance=raw['provenance'], field_V_per_cm=fields.tolist())
    for key, allowed in (('energies_MHz', atomic_levels(config)), ('dipoles_ea0', cavity_links(config))):
        table = raw.get(key, {})
        if not isinstance(table, dict) or set(table)-set(allowed):
            raise ValueError(key + ' references undefined atomic states or cavity links.')
        result[key] = {}
        for name, values in table.items():
            try:
                array = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(key + ' must contain finite numeric arrays.') from exc
            if array.shape != fields.shape or not np.isfinite(array).all():
                raise ValueError(key + ' arrays must match the field grid and be finite.')
            result[key][name] = array.tolist()
    if result['energies_MHz'] and not {'r0', 'r1'} <= set(result['energies_MHz']):
        raise ValueError('Tabulated energies must include both r0 and r1.')
    if not result['energies_MHz'] and not result['dipoles_ea0']:
        raise ValueError('ATOMIC_SPECTRUM requires energy or dipole data.')
    if config['DIRECT_COUPLING'] and result['dipoles_ea0']:
        raise ValueError('Tabulated dipoles require DIRECT_COUPLING=0.')
    park = (config['V_SWAP']/config['GAP_CM']+config['STRAY_FIELD']
            if config['CONSTANT_FIELD'] else config['E_PARK'])
    swap = config['V_SWAP']/config['GAP_CM']+config['STRAY_FIELD']
    epsilon = 1e-12*max(1., abs(fields[0]), abs(fields[-1]))
    if fields[0] > min(park, swap)-config['EFIELD_TOL']+epsilon or fields[-1] < max(park, swap)+config['EFIELD_TOL']-epsilon:
        raise ValueError('Atomic field grid must cover park, entered swap, and electric-field tolerances.')
    return result


@lru_cache(maxsize=32)
def _tables(serialized):
    spec = json.loads(serialized)
    if not spec:
        return {}, {}, None
    x = np.asarray(spec['field_V_per_cm'])
    return (*({name: PchipInterpolator(x, y, extrapolate=False) for name, y in spec[key].items()}
              for key in ('energies_MHz', 'dipoles_ea0')), (float(x[0]), float(x[-1])))


class AtomicModel:
    def __init__(self, config):
        self.config = config
        self.levels, self.links = atomic_levels(config), cavity_links(config)
        spec = config.get('ATOMIC_SPECTRUM', '{}')
        serialized = spec if isinstance(spec, str) else json.dumps(spec, sort_keys=True)
        self.energies, self.dipoles, self.domain = _tables(serialized)

    def _check(self, field):
        values = np.asarray(field)
        epsilon = 1e-12*max(1., *(abs(x) for x in self.domain)) if self.domain else 0.
        if not np.isfinite(values).all() or (self.domain is not None and
                (np.any(values < self.domain[0]-epsilon) or np.any(values > self.domain[1]+epsilon))):
            raise ValueError(f'Field {field} lies outside atomic data domain {self.domain}; extrapolation is disabled.')
        return np.clip(values, *self.domain) if self.domain else values

    def energy(self, name, field, derivative=False):
        field = self._check(field)
        if name in self.energies:
            return self.energies[name](field, nu=int(derivative))
        _, offset, stark = self.levels[name]
        return 2*stark*np.asarray(field) if derivative else offset+stark*np.asarray(field)**2

    def transition(self, field):
        return self.energy('r1', field)-self.energy('r0', field)

    def coupling(self, link, field, derivative=False):
        field = self._check(field)
        c = self.config
        if link in self.dipoles:
            return DIPOLE_TO_MHZ*c['E_ZPF_V_PER_M']*self.dipoles[link](field, nu=int(derivative))
        if derivative:
            return np.zeros_like(np.asarray(field), dtype=float)
        base = (c['G_AC_MHZ'] if c['DIRECT_COUPLING'] else
                DIPOLE_TO_MHZ*c['D_01_EA0']*c['E_ZPF_V_PER_M']*abs(c['DIPOLE_ALIGNMENT']))
        return base*self.links[link]+np.zeros_like(np.asarray(field), dtype=float)

    def resonance(self, reference):
        if not self.energies:
            stark = self.config['K_R1']-self.config['K_R0']
            if stark == 0:
                if not math.isclose(self.config['F_A_ZERO'], self.config['FC'], abs_tol=1e-9, rel_tol=0):
                    raise ValueError('Equal Stark coefficients cannot tune the atom to cavity resonance.')
                return 0.
            square = (self.config['FC']-self.config['F_A_ZERO'])/stark
            if square < 0:
                raise ValueError('The atomic Stark model cannot reach cavity resonance.')
            root = math.sqrt(square)
            self._check(root)
            return root
        p0, p1 = self.energies['r0'], self.energies['r1']
        difference = PPoly(p1.c-p0.c, p0.x, extrapolate=False)
        roots = difference.solve(self.config['FC'], extrapolate=False)
        roots = [float(x) for x in roots if np.isfinite(x) and self.domain[0] <= x <= self.domain[1]]
        # An identically resonant segment is represented by a NaN by PPoly.solve.
        if abs(float(self.transition(reference))-self.config['FC']) < 1e-9:
            return float(reference)
        if not roots:
            raise ValueError('No atomic-cavity resonance exists within the supplied field grid.')
        return min(roots, key=lambda x: abs(x-reference))
