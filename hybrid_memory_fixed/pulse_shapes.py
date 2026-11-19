"""Deterministic pulse envelopes shared by simulation and preview."""
import json
import math
from functools import lru_cache
import numpy as np

CHANNELS = ('optical0', 'optical1', 'flux', 'voltage')


def parse_shapes(value):
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError) as exc:
        raise ValueError('Pulse shapes must be a JSON object.') from exc
    if not isinstance(raw, dict) or set(raw)-set(CHANNELS):
        raise ValueError('Pulse shapes contain an unknown channel.')
    result = {}
    for channel, item in raw.items():
        if not isinstance(item, dict) or set(item)-{'base', 'gaussians', 'noise', 'seed'}:
            raise ValueError(f'{channel}: unknown pulse shape settings.')
        try:
            base, noise = float(item.get('base', 1)), float(item.get('noise', 0))
            seed = int(item.get('seed', 0))
            if isinstance(item.get('seed', 0), bool) or float(item.get('seed', 0)) != seed:
                raise ValueError()
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{channel}: use finite numbers and an integer seed.') from None
        if not math.isfinite(base) or not math.isfinite(noise) or noise < 0 or not 0 <= seed <= 2**32-1:
            raise ValueError(f'{channel}: noise must be nonnegative and seed between 0 and 4294967295.')
        components = item.get('gaussians', [])
        if not isinstance(components, list) or len(components) > 20:
            raise ValueError(f'{channel}: use a list of at most 20 Gaussian peaks.')
        rows = []
        for row in components:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(f'{channel}: each peak requires amplitude, center, width.')
            try:
                amplitude, center, width = map(float, row)
            except (TypeError, ValueError):
                raise ValueError(f'{channel}: peak values must be numbers.') from None
            if not all(math.isfinite(x) for x in (amplitude, center, width)) or not 0 <= center <= 1 or not .01 <= width <= 1:
                raise ValueError(f'{channel}: center must be 0–1 and width 0.01–1.')
            rows.append([amplitude, center, width])
        result[channel] = dict(base=base, gaussians=rows, noise=noise, seed=seed)
    return result


@lru_cache(maxsize=128)
def noise_samples(seed):
    return np.random.default_rng(seed).standard_normal(65)


def pulse_window(t, start, end, shapes, channel):
    """Windowed sin² baseline, added Gaussian bumps, and seeded sampled noise.

    Gaussian and noise contributions are tapered by sin². Values are not
    normalized: changing the shape changes pulse area and potentially peaks.
    """
    if np.isscalar(t):
        if t <= start or t >= end:
            return 0.0
        u = (t-start)/(end-start)
        spec = shapes.get(channel, {})
        value = spec.get('base', 1)
        for amplitude, center, width in spec.get('gaussians', []):
            value += amplitude*math.exp(-.5*((u-center)/width)**2)
        if spec.get('noise', 0):
            position = u*64
            index = min(int(position), 63)
            samples = noise_samples(spec['seed'])
            value += spec['noise']*(samples[index]+(position-index)*(samples[index+1]-samples[index]))
        return math.sin(math.pi*u)**2*value
    x = np.asarray(t, dtype=float)
    u = np.clip((x-start)/(end-start), 0, 1)
    gate = np.sin(np.pi*u)**2
    spec = shapes.get(channel, {})
    value = np.full_like(u, spec.get('base', 1), dtype=float)
    for amplitude, center, width in spec.get('gaussians', []):
        value += amplitude*np.exp(-.5*((u-center)/width)**2)
    if spec.get('noise', 0):
        value += spec['noise']*np.interp(u, np.linspace(0, 1, 65), noise_samples(spec['seed']))
    result = np.where((x > start) & (x < end), gate*value, 0.)
    return float(result) if result.ndim == 0 else result
