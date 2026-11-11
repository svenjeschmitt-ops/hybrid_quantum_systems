"""Structured, nested calculation progress shared by panel subprocesses."""
from contextlib import contextmanager
import json
import math
import os

PREFIX = 'HYBRID_PROGRESS '
_enabled = os.environ.get('HYBRID_SIM_PROGRESS') == '1'
_base = float(os.environ.get('HYBRID_PROGRESS_START', '0'))
_span = float(os.environ.get('HYBRID_PROGRESS_SPAN', '1'))


def enable(value=True):
    global _enabled
    _enabled = bool(value)


def update(fraction, label, complete=False):
    import simulation_stop
    simulation_stop.stage(label)
    if _enabled:
        percent = 100 * max(0, min(1, _base + _span * max(0, min(1, fraction))))
        if not complete:
            percent = min(99, percent)
        print(PREFIX + json.dumps({'percent': percent, 'label': label}), flush=True)


@contextmanager
def scope(start, stop, label):
    global _base, _span
    previous = _base, _span
    _base, _span = _base + _span * start, _span * (stop-start)
    update(0, label)
    try:
        yield
    except BaseException:
        raise
    else:
        update(1, label)
    finally:
        _base, _span = previous


def transfer_stages(config):
    """Reserve work for every extra tolerance trajectory and logical input."""
    axes=('FC_TOL','FT_TOL','FA_TOL','FLUX_TOL','EFIELD_TOL')
    scenarios=3**sum(config.get(key,0)>0 for key in axes)
    total=6+4*(scenarios-1)
    return {name: .95*units/total for name,units in
            (('calibration',1),('selected',2),('storage',3),('basis',6))}


def child_environment(start, stop):
    return {'HYBRID_SIM_PROGRESS': '1' if _enabled else '0',
            'HYBRID_PROGRESS_START': str(_base + _span*start),
            'HYBRID_PROGRESS_SPAN': str(_span*(stop-start))}


def parse_line(line):
    if not line.startswith(PREFIX):
        return None
    try:
        event = json.loads(line[len(PREFIX):])
        percent = float(event['percent'])
        if not math.isfinite(percent) or not 0 <= percent <= 100 or not isinstance(event['label'], str):
            return None
        return {'percent': percent, 'label': event['label']}
    except (ValueError, TypeError, KeyError):
        return None
