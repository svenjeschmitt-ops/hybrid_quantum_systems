"""Bounded multi-control search with full noisy/tolerance validation of the target."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from hybrid_settings import (DEFAULTS, coupling_strengths, calculated_swap_field, calculated_transmon_frequency,
                             derived, resonant_flux, resonant_field, validate, active_delay_keys, pulse_intervals)
from atomic_model import AtomicModel
from simulation_output import real_probability
from assistant_calculations import (MODES, CONTROL_FIELDS, ComparisonCancelled, _score,
                                    evaluate_backend)


class SearchBudget(Exception):
    pass


class StageReached(Exception):
    pass


def requested_target(question):
    """Recognize numeric fidelity goals, including common misspellings and omitted %."""
    if not re.search(r'fid\w*', question, re.I) or not re.search(r'\b(?:target|reach\w*|achiev\w*|at\s+least|below|want|improv\w*|increas\w*|boost\w*|optimi[sz]\w*|tun\w*|adjust\w*|change\w*|set|get)\b', question, re.I):
        return None
    match = re.search(r'(?:at\s+least|target|reach|achieve|want|below)\D{0,30}?(\d+(?:\.\d+)?)\s*(?:%|percent)', question, re.I)
    if not match:
        matches=list(re.finditer(r'(\d+(?:\.\d+)?)\s*(?:%|percent)',question,re.I))
        match=matches[-1] if matches else None
    explicit_percent = match is not None
    if not match:
        # Accept natural follow-ups such as "at least 80 fidility" without
        # treating unrelated numeric device parameters as fidelity goals.
        match = re.search(r'(\d+(?:\.\d+)?)\s+fid\w*\b', question, re.I)
        if not match:
            match = re.search(r'\bfid\w*\b.{0,40}?\b(?:at\s+least|to|of|target)\s+(\d+(?:\.\d+)?)\b', question, re.I)
    if not match:
        return None
    number = float(match.group(1))
    value = number/100 if explicit_percent or number > 1 else number
    if not 0 < value <= 1:
        raise ValueError('Target fidelity must be greater than 0 and at most 100%.')
    return value


def constant_field_requested(question):
    return bool(re.search(
        r'field\b.{0,80}(?:stay|remain|kept|keep|must|need).{0,40}(?:constant|fixed|unchanged|exact)|'
        r'(?:keep|hold|fix).{0,40}field\b.{0,40}(?:constant|fixed|unchanged|exact)|'
        r'(?:keep|hold).{0,30}(?:constant|fixed).{0,20}field', question, re.I))


def locked_controls(config):
    """Constant entered controls are fixed across search trials, not just in time."""
    keys=[]
    if config['CONSTANT_FIELD']:
        keys+=['CONSTANT_FIELD','V_SWAP','GAP_CM','STRAY_FIELD','E_PARK']
    if config['CONSTANT_FLUX']:
        keys+=['CONSTANT_FLUX','PHI_SWAP','PHI_PARK']
    return {key:config[key] for key in keys}


def field_diagnostics(config):
    if not config['CONSTANT_FIELD']:
        return None
    electric=calculated_swap_field(config)
    frequency=float(AtomicModel(config).transition(electric))
    return dict(field_V_per_cm=electric,voltage_V=config['V_SWAP'],
                atomic_transition_MHz=frequency,cavity_MHz=config['FC'],
                atomic_detuning_MHz=frequency-config['FC'])


def physical_search_space(c, mode, vary_cavity=False):
    """Search applied controls directly when a quadratic inversion is unavailable."""
    import numpy as np
    atom = AtomicModel(c)
    entries = []
    def add(key, low, high, logarithmic=False):
        entries.append((key, (math.log(low), math.log(high)) if logarithmic else (low, high), logarithmic))
    if vary_cavity:
        low = max(1., calculated_transmon_frequency(c,.5)+201, .65*c['FC'])
        high = min(calculated_transmon_frequency(c,0)-201, 1.35*c['FC'])
        if mode.startswith('Simultaneous cavity resonance:') and not c['CONSTANT_FIELD']:
            if c['K_R1'] < c['K_R0']:
                high = min(high, c['F_A_ZERO']-1e-6)
            elif c['K_R1'] > c['K_R0']:
                low = max(low, c['F_A_ZERO']+1e-6)
            else:
                raise ValueError('A variable cavity search needs a tunable atomic transition or a constant field.')
        add('FC', low, high)
    if mode.startswith('Simultaneous cavity resonance:'):
        add('SIM_HOLD_SCALE', .4, 2.5, True)
        if not (c['CONSTANT_FLUX'] and c['CONSTANT_FIELD']):
            add('SIM_RAMP', c['SIM_RAMP']*.2, c['SIM_RAMP']*5, True)
    else:
        if not c['CONSTANT_FLUX']:
            add('PHI_SWAP', max(0., c['PHI_SWAP']-.08), min(.5, c['PHI_SWAP']+.08))
        add('TC_PULSE_SCALE', .4, 25., True)
        if not c['CONSTANT_FIELD']:
            field = calculated_swap_field(c)
            low, high = (field-max(1.,abs(field)*.5), field+max(1.,abs(field)*.5))
            low, high = low+c['EFIELD_TOL'], high-c['EFIELD_TOL']
            add('V_SWAP', c['GAP_CM']*(low-c['STRAY_FIELD']), c['GAP_CM']*(high-c['STRAY_FIELD']))
        add('AC_PULSE_SCALE', .4, 25., True)
    if not mode.startswith('Simultaneous cavity resonance:'):
        timing = derived(c)
        for key in active_delay_keys(mode):
            if c[key] is not None:
                span = max(abs(c[key])*2,
                           1000*(timing['transmon_pulse_duration_us']+timing['atom_pulse_duration_us']+c['GAP']))
                add(key, -span, span)
    add('OMEGA', c['OMEGA']*.5, c['OMEGA']*2, True)
    add('OMEGA_1', c['OMEGA_1']*.5, c['OMEGA_1']*2, True)
    add('GAP', 0., max(.03, c['GAP']*2))
    if any(low >= high for _, (low, high), _ in entries):
        raise ValueError('No nonempty physical-control search interval is available.')
    keys = [key for key, _, _ in entries]
    bounds = [bound for _, bound, _ in entries]
    def encode(values):
        return np.clip([math.log(values[key]) if log else values[key] for key, _, log in entries],
                       [low for low, high in bounds], [high for low, high in bounds])
    def decode(values):
        return validate({**c, **{key: float(math.exp(x) if log else x)
                                for (key, _, log), x in zip(entries, values)}})
    seed = dict(c)
    try:
        if 'PHI_SWAP' in keys:
            seed['PHI_SWAP'] = resonant_flux(c)
        if 'V_SWAP' in keys:
            seed['V_SWAP'] = c['GAP_CM']*(resonant_field(c)-c['STRAY_FIELD'])
    except ValueError:
        pass
    return keys, bounds, encode(c), [encode(c), encode(seed)], decode


def search_space(config, mode, vary_cavity=False):
    """Peak detunings and log durations avoid ill-scaled flux/voltage coordinates."""
    import numpy as np
    c = validate(config)
    if mode not in MODES:
        raise ValueError('Unsupported search output mode.')
    if (any(c[key] is not None for key in active_delay_keys(mode))
            and not mode.startswith('Simultaneous cavity resonance:')):
        return physical_search_space(c, mode, vary_cavity)
    if vary_cavity:
        keys,bounds,current,seeds,_=search_space(c,mode,False)
        fmin=calculated_transmon_frequency(c,.5)+201
        fmax=calculated_transmon_frequency(c,0)-201
        atomic=(c['F_A_ZERO']+(c['K_R1']-c['K_R0'])*calculated_swap_field(c)**2
                if c['CONSTANT_FIELD'] else c['FC'])
        centers=[c['FC']]
        if atomic>0:
            centers.insert(0,atomic)
        if c['CONSTANT_FLUX']:
            centers.append(calculated_transmon_frequency(c,c['PHI_SWAP']))
        lower=max(1.,fmin,.65*min(centers))
        upper=min(fmax,1.35*max(centers))
        if not c['CONSTANT_FIELD']:
            stark=c['K_R1']-c['K_R0']
            if stark<0:
                upper=min(upper,c['F_A_ZERO']-1e-3)
            elif stark>0:
                lower=max(lower,c['F_A_ZERO']+1e-3)
        if lower>=upper:
            raise ValueError('No positive cavity-frequency search interval is reachable with this transmon and atomic model.')
        def decode(x):
            shifted=dict(c,FC=float(x[0]))
            inner_keys,_,_,_,inner_decode=search_space(shifted,mode,False)
            if inner_keys!=keys:
                raise ValueError('Active search controls changed unexpectedly.')
            return inner_decode(x[1:])
        frequency_seeds=[]
        for center in centers:
            value=float(np.clip(center,lower,upper))
            try:
                _,_,_,local_seeds,_=search_space(dict(c,FC=value),mode,False)
                frequency_seeds.append(np.r_[value,local_seeds[0]])
            except ValueError:
                pass
        frequency_seeds += [np.r_[float(np.clip(c['FC'],lower,upper)),seed] for seed in seeds]
        return ['FC',*keys],[(lower,upper),*bounds],np.r_[float(np.clip(c['FC'],lower,upper)),current],frequency_seeds,decode
    if mode.startswith('Simultaneous cavity resonance:'):
        keys = ['SIM_HOLD_SCALE', 'OMEGA', 'OMEGA_1', 'GAP']
        bounds = [(math.log(.4), math.log(2.5)),
                  (math.log(c['OMEGA']*.5), math.log(c['OMEGA']*2)),
                  (math.log(c['OMEGA_1']*.5), math.log(c['OMEGA_1']*2)), (0, max(.03, c['GAP']*2))]
        if not (c['CONSTANT_FLUX'] and c['CONSTANT_FIELD']):
            keys.append('SIM_RAMP')
            bounds.append((math.log(c['SIM_RAMP']*.2), math.log(c['SIM_RAMP']*5)))
        seed=dict(c)
        linear={'GAP','PHI_SWAP','V_SWAP'}
        def encode(values):
            x=np.array([values[key] if key in linear else math.log(values[key]) for key in keys])
            return np.clip(x,[a for a,b in bounds],[b for a,b in bounds])
        def decode(x):
            return validate({**c, **{key:float(value) if key in linear else math.exp(value) for key,value in zip(keys,x)}})
        return keys, bounds, encode(c), [encode(seed),encode(c)], decode
    stark = c['K_R1']-c['K_R0']
    if stark == 0 and not c['CONSTANT_FIELD']:
        raise ValueError('A tunable differential atomic Stark shift is required for this sequential search.')
    park_t = calculated_transmon_frequency(c,c['PHI_PARK'])-c['FC']
    park_a = c['F_A_ZERO']+stark*c['E_PARK']**2-c['FC']
    def detuning_bounds(park):
        return (-200.,-.05) if park >= 0 else (.05,200.)
    tbound, abound = detuning_bounds(park_t), detuning_bounds(park_a)
    tbound = (max(tbound[0],calculated_transmon_frequency(c,.5)-c['FC']+1),
              min(tbound[1],calculated_transmon_frequency(c,0)-c['FC']-1))
    # Trim the atomic interval to real fields.
    if stark < 0:
        abound = (abound[0],min(abound[1],c['F_A_ZERO']-c['FC']-1e-6))
    else:
        abound = (max(abound[0],c['F_A_ZERO']-c['FC']+1e-6),abound[1])
    if (not c['CONSTANT_FLUX'] and tbound[0]>=tbound[1]) or (not c['CONSTANT_FIELD'] and abound[0]>=abound[1]):
        raise ValueError('No reachable resonance-crossing interval exists within the search bounds.')
    keys = ['PHI_SWAP','TC_PULSE_SCALE','V_SWAP','AC_PULSE_SCALE','GAP','OMEGA','OMEGA_1']
    bounds = [tbound,(math.log(.4),math.log(25)),abound,(math.log(.4),math.log(25)),(0.,max(.03,c['GAP']*2)),(math.log(c['OMEGA']*.5),math.log(c['OMEGA']*2)),(math.log(c['OMEGA_1']*.5),math.log(c['OMEGA_1']*2))]
    def encode(values):
        return np.array([calculated_transmon_frequency(values,values['PHI_SWAP'])-values['FC'],
                         math.log(values['TC_PULSE_SCALE']),
                         values['F_A_ZERO']+stark*calculated_swap_field(values)**2-values['FC'],
                         math.log(values['AC_PULSE_SCALE']), values['GAP'], math.log(values['OMEGA']), math.log(values['OMEGA_1'])])
    def decode(x):
        phi = c['PHI_SWAP'] if c['CONSTANT_FLUX'] else resonant_flux(dict(c,FC=c['FC']+float(x[0])))
        field = calculated_swap_field(c) if c['CONSTANT_FIELD'] else math.sqrt((c['FC']+float(x[2])-c['F_A_ZERO'])/stark)
        return validate(dict(c,PHI_SWAP=phi,TC_PULSE_SCALE=math.exp(x[1]),
                             V_SWAP=c['V_SWAP'] if c['CONSTANT_FIELD'] else c['GAP_CM']*(field-c['STRAY_FIELD']),AC_PULSE_SCALE=math.exp(x[3]),GAP=float(x[4]),OMEGA=math.exp(x[5]),OMEGA_1=math.exp(x[6])))
    def clip(x):
        return np.clip(x,[a for a,b in bounds],[b for a,b in bounds])
    old = derived(DEFAULTS)
    seed = np.array([old['transmon_swap_detuning_MHz'],
        math.log(DEFAULTS['TC_PULSE_SCALE']*math.sqrt(max(.1,abs(park_t)/max(1,abs(old['transmon_park_frequency_MHz']-DEFAULTS['FC']))))),
        old['atom_swap_detuning_MHz'],
        math.log(DEFAULTS['AC_PULSE_SCALE']*math.sqrt(max(.1,abs(park_a)/150))),c['GAP'],math.log(c['OMEGA']),math.log(c['OMEGA_1'])])
    seeds = [clip(seed),clip(encode(c)),clip([old['transmon_swap_detuning_MHz'],math.log(c['TC_PULSE_SCALE']),
                                           old['atom_swap_detuning_MHz'],math.log(c['AC_PULSE_SCALE']),c['GAP'],math.log(c['OMEGA']),math.log(c['OMEGA_1'])])]
    if c['CONSTANT_FIELD'] or c['CONSTANT_FLUX']:
        original_decode=decode
        positions=[i for i,key in enumerate(keys) if key not in locked_controls(c) and key not in ('GAP','OMEGA','OMEGA_1')]
        held=encode(c)
        active_keys=[keys[i] for i in positions]
        active_bounds=[bounds[i] for i in positions]
        # With a fixed field/flux, optimize optical timing too; never alter the field itself.
        active_keys+=['OMEGA','OMEGA_1','GAP']
        active_bounds+=[(math.log(c['OMEGA']*.5),math.log(c['OMEGA']*2)),(math.log(c['OMEGA_1']*.5),math.log(c['OMEGA_1']*2)),(0,max(.03,c['GAP']*2))]
        def compact(x):
            return np.r_[np.asarray(x)[positions],math.log(c['OMEGA']),math.log(c['OMEGA_1']),c['GAP']]
        active_seeds=[compact(x) for x in seeds]
        def decode_active(x):
            full=held.copy();full[positions]=x[:len(positions)]
            candidate=original_decode(full)
            candidate.update(OMEGA=math.exp(x[-3]),OMEGA_1=math.exp(x[-2]),GAP=float(x[-1]))
            return validate(candidate)
        return active_keys,active_bounds,compact(held),active_seeds,decode_active
    return keys,bounds,clip(encode(c)),seeds,decode


def population_seed(config):
    """Find inexpensive two-level pulse seeds; full multilevel evolution verifies them.

    This surrogate is used only to initialize the search, never as fidelity
    evidence. Midpoint propagators retain the actual rounded control waveform.
    """
    import numpy as np
    from scipy.optimize import differential_evolution
    c = validate(config)
    atom = AtomicModel(c)
    grid = (np.arange(1024)+.5)/1024
    window = np.sin(np.pi*grid)**2
    result = dict(c)
    for subsystem in ('transmon', 'atom'):
        if subsystem == 'transmon' and c['CONSTANT_FLUX']:
            continue
        if subsystem == 'atom' and c['CONSTANT_FIELD']:
            continue
        def candidate(x):
            if subsystem == 'transmon':
                phi = resonant_flux(dict(c, FC=c['FC']+float(x[0])))
                values = dict(result, PHI_SWAP=phi)
                fields = c['PHI_PARK']+(phi-c['PHI_PARK'])*window
                ej = c['EJ_MAX']*np.sqrt(np.cos(np.pi*fields)**2+c['ASYMMETRY']**2*np.sin(np.pi*fields)**2)
                delta = np.sqrt(8*c['EC']*ej)-c['EC']-c['FC']
                rate = coupling_strengths(values)['transmon_coupling_MHz']
            else:
                field = np.sqrt((c['FC']+float(x[0])-c['F_A_ZERO'])/(c['K_R1']-c['K_R0']))
                values = dict(result, V_SWAP=c['GAP_CM']*(field-c['STRAY_FIELD']))
                fields = c['E_PARK']+(field-c['E_PARK'])*window
                delta = atom.transition(fields)-c['FC']
                rate = coupling_strengths(values)['atom_coupling_MHz']
            scale = float(np.exp(x[1]))
            duration = scale*(2*c['RAMP']+1/(4*rate))
            dt = duration/len(grid)
            g = 2*np.pi*rate
            z = np.pi*delta
            omega = np.hypot(g,z)
            cs, sn = np.cos(omega*dt), -1j*np.sin(omega*dt)/omega
            a, b = 1.+0j, 0j
            for co, si, zi in zip(cs, sn, z):
                a, b = co*a+si*(zi*a+g*b), co*b+si*(g*a-zi*b)
            key = 'TC_PULSE_SCALE' if subsystem == 'transmon' else 'AC_PULSE_SCALE'
            values[key] = scale
            return abs(b)**2, values
        def objective(x):
            try:
                return 1-candidate(x)[0]
            except ValueError:
                return 1.
        fit = differential_evolution(objective, [(-200.,-.05),(math.log(.4),math.log(25.))],
                                     popsize=8,maxiter=45,tol=1e-6,seed=19,polish=True)
        _, result = candidate(fit.x)
    # The reverse optical arm uses the same carrier convention. Seed an
    # integer relative carrier cycle across the optical pulse so write/read
    # superpositions can share a high-fidelity phase without output correction.
    detuning = abs(float(atom.transition(c['E_PARK']))-c['FC'])
    cycles = max(1, round(detuning/c['OMEGA']))
    omega = detuning/cycles
    if c['OMEGA']*.5 <= omega <= c['OMEGA']*2:
        result['OMEGA'] = omega
    return validate(result)


def load_backend(config, directory):
    """Fresh operators and pulse timings per point in the isolated worker."""
    path=Path(directory)/'screen_config.json'
    path.write_text(json.dumps(validate(config)))
    previous=sys.argv
    sys.argv=['backend.py','--config',str(path)]
    try:
        name='_assistant_search_backend'
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('backend.py'))
        b=importlib.util.module_from_spec(spec)
        sys.modules[name]=b
        spec.loader.exec_module(b)
        return b
    finally:
        sys.argv=previous


_COHERENT_STAGES = {}
_COHERENT_PULSES = {}


def sequential_screen(b, mode, stage, amplitudes=False):
    """Exact unitary stage composition in the invariant zero/one-excitation space.

    Idle intervals use matrix exponentials. Pulse propagators can be reused
    while tuning GAP because each optical carrier has a local pulse origin.
    No atomic levels within these invariant sectors are discarded.
    """
    import numpy as np
    import qutip as qt
    from scipy.linalg import expm
    atomic = {name: level[0] for name, level in b._ATOMIC_MODEL.levels.items()}
    atomic.update(h0=0,h1=1,lost=0,optical_p0=0,optical_p1=1)
    order = list(b.ATOM_STATES)
    indices = [((t*b.NC+c)*b.NA+a) for t in range(b.NT) for c in range(b.NC)
               for a,name in enumerate(order) if t+c+atomic[name] <= 1]
    data = np.eye(b.NT*b.NC*b.NA)[:,indices]
    projection = qt.Qobj(data, dims=[[b.NT,b.NC,b.NA],[len(indices)]])
    settings = {k:v for k,v in b._SETTINGS.items() if k not in ('GAP','WRITE_PULSE_DELAY_NS','READ_PULSE_DELAY_NS')}
    scores = []
    logical_amplitudes = {}
    for direction in (MODES[mode] if stage == 'joint' else ('write',)):
        if direction == 'write':
            h = b.H
            intervals = [b.PREP,b.TC,b.AC,b.MAP]
        else:
            h,_,stages,*_ = b.retrieval_model()
            intervals = [pair for _,pair in stages]
        key = (json.dumps(settings,sort_keys=True), direction)
        if key not in _COHERENT_STAGES:
            reduced = projection.dag()*h*projection
            options=dict(atol=1e-11,rtol=1e-9,method='dop853',max_step=b.MAX_STEP,nsteps=100000)
            pulses = []
            pulse_settings = {k:v for k,v in settings.items() if k not in ('TC_PULSE_SCALE','AC_PULSE_SCALE')}
            kinds = ('optical0','tc','ac','optical01') if direction == 'write' else ('optical01','ac','tc','optical0')
            for kind,(start,end) in zip(kinds,intervals):
                pulse_key = (json.dumps(pulse_settings,sort_keys=True),kind,round(end-start,14))
                if pulse_key not in _COHERENT_PULSES:
                    if len(_COHERENT_PULSES) >= 64:
                        _COHERENT_PULSES.pop(next(iter(_COHERENT_PULSES)))
                    _COHERENT_PULSES[pulse_key] = qt.propagator(reduced,[start,end],options=options)[-1].full()
                pulses.append(_COHERENT_PULSES[pulse_key])
            parked = reduced(intervals[-1][1]).full()
            if len(_COHERENT_STAGES) >= 16:
                _COHERENT_STAGES.pop(next(iter(_COHERENT_STAGES)))
            _COHERENT_STAGES[key] = pulses,parked
        pulses, parked = _COHERENT_STAGES[key]
        if direction == 'write':
            initial = qt.tensor((b.g+b.e).unit() if stage=='joint' else b.e,
                                qt.basis(b.NC,0),b.h0)
        else:
            initial = b.retrieval_initial(1,1)
        state = (projection.dag()*initial).full().ravel()
        if direction == 'read':
            state = expm(-1j*parked*b.STORAGE)@state
        for index,pulse in enumerate(pulses):
            state = pulse@state
            if stage == 'tc' and index == 1:
                break
            if index < len(pulses)-1:
                idle_time = intervals[index+1][0]-intervals[index][1]
                state = expm(-1j*parked*idle_time)@state
        if stage != 'joint':
            target = qt.tensor(b.g,qt.basis(b.NC,1),b.r0) if stage=='tc' else qt.tensor(b.g,qt.basis(b.NC,0),b.h1)
            scores.append(abs(np.vdot((projection.dag()*target).full().ravel(),state))**2)
            continue
        if direction == 'write':
            state = expm(-1j*parked*b.STORAGE)@state
        zero = qt.tensor(b.g,qt.basis(b.NC,0),b.h0)
        one = qt.tensor(b.g,qt.basis(b.NC,0),b.h1) if direction=='write' else qt.tensor(b.e,qt.basis(b.NC,0),b.h0)
        a0 = np.sqrt(2)*np.vdot((projection.dag()*zero).full().ravel(),state)
        a1 = np.sqrt(2)*np.vdot((projection.dag()*one).full().ravel(),state)
        logical_amplitudes[direction] = (a0,a1)
        scores.extend([abs(a0)**2,abs(a1)**2,abs(a0+a1)**2/4])
    return logical_amplitudes if amplitudes else float(min(scores))


def screen(b, mode, stage='joint'):
    """Noiseless endpoint screening only; never used as final target evidence."""
    import numpy as np
    import qutip as qt
    options=dict(store_states=True,atol=1e-11,rtol=1e-9,method='dop853',max_step=b.MAX_STEP,nsteps=100000)
    if not mode.startswith('Simultaneous cavity resonance:'):
        read_tc,read_ac=pulse_intervals(b._SETTINGS,max(b.T_OPT,b.T_OPT_1)+b.GAP,b.TC_DURATION,b.AC_DURATION,'read')
        separated = all(b.TC[1] <= b.AC[0]+1e-12 if direction=='write' else read_ac[1] <= read_tc[0]+1e-12
                        for direction in MODES[mode])
        if separated:
            return sequential_screen(b, mode, stage)
    if mode.startswith('Simultaneous cavity resonance:'):
        from simultaneous_coupling import model
        scores=[]
        for direction in MODES[mode]:
            h,_,info,*_=model(b,False,direction)
            times=np.unique([0.,*[x for pair in info['stages'].values() for x in pair]])
            initial=qt.tensor((b.g+b.e).unit(),qt.basis(b.NC,0),b.h0) if direction=='write' else b.retrieval_initial(1,1)
            if direction=='read':
                initial=(-1j*h(0)*b.STORAGE).expm()*initial
            final=qt.sesolve(h,initial,times,options=options).states[-1]
            if direction=='write':
                final=(-1j*h(times[-1])*b.STORAGE).expm()*final
            zero=qt.tensor(b.g,qt.basis(b.NC,0),b.h0)
            one=qt.tensor(b.g,qt.basis(b.NC,0),b.h1) if direction=='write' else qt.tensor(b.e,qt.basis(b.NC,0),b.h0)
            a0,a1=np.sqrt(2)*zero.overlap(final),np.sqrt(2)*one.overlap(final)
            scores.extend([abs(a0)**2,abs(a1)**2,abs(a0+a1)**2/4])
        return min(scores)
    scores=[]
    if stage in ('tc','ac') or 'write' in MODES[mode]:
        end=b.TC[1] if stage=='tc' else b.MAP[1]
        times=np.unique([0.,*[x for pair in (b.PREP,b.TC,b.AC,b.MAP) for x in pair if x<=end]])
        initial=qt.tensor((b.g+b.e).unit() if stage=='joint' else b.e,qt.basis(b.NC,0),b.h0)
        final=qt.sesolve(b.H,initial,times,options=options).states[-1]
        target=qt.tensor(b.g,qt.basis(b.NC,1),b.r0) if stage=='tc' else qt.tensor(b.g,qt.basis(b.NC,0),b.h1)
        if stage == 'joint':
            final = (-1j*b.H(b.MAP[1])*b.STORAGE).expm()*final
            zero = qt.tensor(b.g,qt.basis(b.NC,0),b.h0)
            a0, a1 = np.sqrt(2)*zero.overlap(final), np.sqrt(2)*target.overlap(final)
            scores.extend([abs(a0)**2, abs(a1)**2, abs(a0+a1)**2/4])
        else:
            scores.append(real_probability(qt.expect(target.proj(),final)))
    if stage=='joint' and 'read' in MODES[mode]:
        h,_,stages,*_=b.retrieval_model()
        times=np.unique([0.,*[x for _,pair in stages for x in pair]])
        initial=b.hold_memory(b.retrieval_initial(1,1),noisy=False)
        final=qt.sesolve(h,initial,times,options=options).states[-1]
        zero=qt.tensor(b.g,qt.basis(b.NC,0),b.h0)
        one=qt.tensor(b.e,qt.basis(b.NC,0),b.h0)
        a0, a1 = np.sqrt(2)*zero.overlap(final), np.sqrt(2)*one.overlap(final)
        scores.extend([abs(a0)**2, abs(a1)**2, abs(a0+a1)**2/4])
    return min(scores)


def optimize_controls(config, mode, target, directory, max_evaluations=320, evaluator=None, vary_cavity=False):
    import numpy as np
    from scipy.optimize import minimize, differential_evolution
    keys,bounds,current,seeds,decode=search_space(config,mode,vary_cavity)
    offset = int(keys[0] == 'FC')
    seeded = keys[offset:offset+4] == ['PHI_SWAP','TC_PULSE_SCALE','V_SWAP','AC_PULSE_SCALE'] and len(keys) == offset+7
    if seeded:
        seed = population_seed(config)
        coordinates = [calculated_transmon_frequency(seed,seed['PHI_SWAP'])-seed['FC'],
                       math.log(seed['TC_PULSE_SCALE']),
                       float(AtomicModel(seed).transition(calculated_swap_field(seed)))-seed['FC'],
                       math.log(seed['AC_PULSE_SCALE']), seed['GAP'], math.log(seed['OMEGA']), math.log(seed['OMEGA_1'])]
        seeds.insert(0, np.array(([seed['FC']] if offset else [])+coordinates))
    cache,trace,candidates={},[],[]
    best_score,best_x=-1.,current.copy()
    def update(stage):
        state=dict(stage=stage,evaluations=len(trace),budget=max_evaluations,
                   best_noiseless_score=best_score,target=target,parameters=keys)
        path=Path(directory)/'search_state.json'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state));temporary.replace(path)
    def evaluate(x,stage):
        nonlocal best_score,best_x
        code=(stage,*map(float,x))
        if code in cache:
            return cache[code]
        if len(trace)>=max_evaluations:
            raise SearchBudget()
        try:
            c=decode(x)
        except ValueError as exc:
            cache[code]=0.
            trace.append(dict(stage=stage,coordinates=list(map(float,x)),invalid=str(exc)))
            update(stage)
            return 0.
        score=float(evaluator(c,stage) if evaluator else screen(load_backend(c,directory),mode,stage))
        if not math.isfinite(score) or not -1e-6<=score<=1+1e-6:
            raise ValueError('Invalid screening result.')
        cache[code]=score
        trace.append(dict(stage=stage,settings={k:c[k] for k in keys},score=score))
        if stage=='joint':
            candidates.append((score,x.copy()))
            if score>best_score:
                best_score,best_x=score,x.copy()
        update(stage)
        return score
    def local(start,indices,stage,goal,budget):
        best=[-1.,start.copy()]
        def objective(values):
            x=start.copy();x[indices]=values
            score=evaluate(x,stage)
            if score>best[0]:
                best[:]=[score,x.copy()]
            if score>=goal:
                raise StageReached()
            return 1-score
        try:
            objective(start[indices])
            minimize(objective,start[indices],method='Powell',bounds=[bounds[i] for i in indices],
                     options=dict(maxfev=budget,xtol=1e-3,ftol=1e-6))
        except StageReached:
            pass
        return best[1]
    goal=min(.9995,target+.12)
    try:
        for seed in (seeds[:1] if seeded else seeds):
            evaluate(seed,'joint')
        if not mode.startswith('Simultaneous cavity resonance:') and keys[:4]==['PHI_SWAP','TC_PULSE_SCALE','V_SWAP','AC_PULSE_SCALE'] and best_score<goal:
            seed=seeds[0]
            tc=local(seed,[0,1],'tc',.985,80)
            ac=local(tc,[2,3],'ac',max(.95,goal),90)
            evaluate(ac,'joint')
            if 'GAP' in keys:
                index=keys.index('GAP')
                rate=abs(calculated_transmon_frequency(config,config['PHI_PARK'])-config['FC'])+abs(float(AtomicModel(config).transition(config['E_PARK']))-config['FC'])
                period=1/max(1.,rate)
                for gap in np.linspace(max(0.,config['GAP']-period),config['GAP']+period,33):
                    trial=ac.copy();trial[index]=np.clip(gap,*bounds[index])
                    evaluate(trial,'joint')
            if best_score<goal:
                other=local(best_x.copy(),list(range(len(keys))),'joint',goal,100)
                evaluate(other,'joint')
        elif best_score<goal:
            local(best_x.copy(),list(range(len(keys))),'joint',goal,110)
        if best_score<goal and len(trace)<max_evaluations:
            def objective(x):
                value=evaluate(x,'joint')
                if value>=goal:
                    raise StageReached()
                return 1-value
            try:
                differential_evolution(objective,bounds,popsize=5,maxiter=20,polish=False,
                                       seed=7,x0=best_x,atol=1e-4)
            except StageReached:
                pass
    except SearchBudget:
        pass
    candidates.sort(key=lambda item:item[0],reverse=True)
    unique=[]
    for score,x in candidates:
        c=decode(x)
        if not any(c==old['settings'] for old in unique):
            unique.append(dict(settings=c,noiseless_score=score))
    result=dict(parameters=keys,bounds=[list(pair) for pair in bounds],evaluations=len(trace),
                screening_goal=goal,trace=trace,candidates=unique[:3])
    (Path(directory)/'screening.json').write_text(json.dumps(result,allow_nan=False))
    return result


def validate_search_result(result, config, mode, target, vary_cavity=None):
    c=validate(config)
    if result['mode']!=mode or result['target']!=target or result['baseline']['settings']!=c:
        raise ValueError('Search evidence does not match the captured request.')
    best=result['best']
    if validate(best['settings'])!=best['settings']:
        raise ValueError('Search candidate settings are not canonical.')
    altered={k for k in c if best['settings'][k]!=c[k]}
    frequency_free=bool(result.get('vary_cavity',False))
    if vary_cavity is not None and frequency_free!=bool(vary_cavity):
        raise ValueError('Cavity-frequency search permissions do not match the request.')
    space=search_space(c,mode,frequency_free)
    allowed=set(space[0])
    if frequency_free and not space[1][0][0]<=best['settings']['FC']<=space[1][0][1]:
        raise ValueError('Found cavity frequency is outside the declared search interval.')
    locks=locked_controls(c)
    if any(best['settings'][key]!=value for key,value in locks.items()):
        raise ValueError('Search changed a locked constant field or flux control.')
    if altered-allowed:
        raise ValueError('Search changed fixed scientific assumptions.')
    before,after=_score(result['baseline'],mode),_score(best,mode)
    if best['mode']!=mode:
        raise ValueError('Search candidate mode mismatch.')
    diagnostic=field_diagnostics(c)
    model_valid=diagnostic is None or diagnostic['atomic_transition_MHz']>0
    result=dict(result,vary_cavity=frequency_free,cavity_search_bounds_MHz=list(space[1][0]) if frequency_free else None,locked_controls=locks,field_diagnostics=field_diagnostics(best['settings']),model_valid=model_valid,baseline_score=before,best_score=after,target_reached=model_valid and after>=target,
                changes={k:best['settings'][k] for k in sorted(altered)})
    return result


def search_answer(result, config, mode, target):
    result=validate_search_result(result,config,mode,target)
    lines=[f"Target: **{target:.2%}** minimum joint fidelity for logical 0, 1 and + across the selected directions.",
           f"Validated score: **{result['baseline_score']:.4%} → {result['best_score']:.4%}**.",
           ('Storage, noise, basis and tolerance bounds were held fixed; cavity frequency was allowed to vary.' if result['vary_cavity'] else 'Storage, noise, cavity frequency, basis and tolerance bounds were held fixed.'),'',
           '| Parameter | Current | Found |','|---|---:|---:|']
    for key in result.get('searched_parameters',[]):
        lines.append(f"| {key} | {config[key]} | {result['best']['settings'][key]} |")
    if result['locked_controls']:
        lines+=['','Locked controls preserved: '+', '.join(f'{key}={value}' for key,value in result['locked_controls'].items())+'.']
    diagnostic=result['field_diagnostics']
    if diagnostic:
        lines+=['',f"Fixed nominal field: {diagnostic['field_V_per_cm']:.6g} V/cm. The implemented Stark model gives atomic transition {diagnostic['atomic_transition_MHz']:.6g} MHz, versus cavity {diagnostic['cavity_MHz']:.6g} MHz (detuning {diagnostic['atomic_detuning_MHz']:+.6g} MHz)."]
        if diagnostic['atomic_transition_MHz']<=0:
            lines+=['At this field the quadratic selected-level model predicts inverted atomic level ordering. Recheck the physical level scheme and Stark-model validity; the search does not change these device assumptions.']
    lines+=['','| Direction | Logical 0 | Logical 1 | Logical + |','|---|---:|---:|---:|']
    for direction,data in result['best']['directions'].items():
        lines.append(f"| {direction} | "+' | '.join(f'{v:.4%}' for v in data['sampled_logical_lows'])+' |')
    if result['target_reached']:
        lines+=['','**Target reached in the full noisy simulation**, including storage and configured tolerance samples.']
    elif not result['model_valid']:
        lines+=['','**Target not certified at this fixed field.** The model predicts a nonpositive atomic transition, so optimization was skipped. No field settings were changed and no proposal is offered.']
    else:
        lines+=['','**Target not reached within this search budget.** The best tested result remains below the request; no target-reaching proposal is offered.']
    lines+=['','Screening used noiseless endpoint populations; the reported fidelities are from full noisy validation. '
            'This is a bounded search, not a global optimum or a guarantee for all qubit inputs or unsampled errors.',
            'Numerical evidence: '+result.get('source','local search')]
    if result['target_reached'] and result['changes']:
        lines+=['','```panel-settings',json.dumps(dict(changes=result['changes'],evidence='measured simulation result',
            reason=f"Multi-parameter search validated minimum fidelity {result['best_score']:.6%}, meeting target {target:.2%}.")),'```']
    return '\n'.join(lines)


def worker(config, mode, target, directory, max_evaluations=320, vary_cavity=False):
    directory=Path(directory)
    def validate_point(c,label):
        folder=directory/label;folder.mkdir(exist_ok=True)
        state=directory/'search_state.json'
        state.write_text(json.dumps(dict(stage='Full noisy validation: '+label,evaluations=0)))
        b=load_backend(c,directory)
        return evaluate_backend(b,mode,folder)
    baseline=validate_point(config,'baseline')
    best=baseline
    screening=dict(parameters=search_space(config,mode,vary_cavity)[0],evaluations=0,candidates=[])
    diagnostic=field_diagnostics(config)
    model_valid=diagnostic is None or diagnostic['atomic_transition_MHz']>0
    if model_valid and _score(baseline,mode)<target:
        screening=optimize_controls(config,mode,target,directory,max_evaluations,vary_cavity=vary_cavity)
        for index,candidate in enumerate(screening['candidates']):
            point=validate_point(candidate['settings'],f'candidate_{index}')
            if _score(point,mode)>_score(best,mode):
                best=point
            if _score(best,mode)>=target:
                break
    result=dict(mode=mode,target=target,vary_cavity=vary_cavity,baseline=baseline,best=best,
                searched_parameters=screening['parameters'],screening_evaluations=screening['evaluations'])
    return validate_search_result(result,config,mode,target)


def run_target_search(config, mode, target, cancelled, project_root, on_progress=None, timeout=2400, vary_cavity=False):
    from codex_assistant import _stop
    config=validate(config)
    if not 0<target<=1:
        raise ValueError('Target fidelity must be greater than 0 and at most 100%.')
    search_space(config,mode,vary_cavity)
    if cancelled.is_set():
        raise ComparisonCancelled()
    root=Path(project_root)/'output'/'assistant_searches';root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='search-',dir=root))
    conf=folder/'config.json';conf.write_text(json.dumps(config))
    output=folder/'result.json'
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(conf),
             '--output',str(output),'--mode',mode,'--target',str(target)]
    if vary_cavity:
        command.append('--vary-cavity')
    env={**os.environ,'MPLBACKEND':'Agg','HYBRID_SIM_PROGRESS':'0'};env.pop('HYBRID_STOP_FILE',None)
    with (folder/'search.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name=='posix')
        start,last_notice=time.monotonic(),-1
        try:
            while True:
                if cancelled.is_set():
                    raise ComparisonCancelled()
                elapsed=int(time.monotonic()-start)
                if elapsed>timeout:
                    raise RuntimeError('Target search timed out. Partial evidence: '+str(folder))
                if on_progress and elapsed!=last_notice:
                    last_notice=elapsed
                    try:
                        state=json.loads((folder/'search_state.json').read_text())
                        message=f"{state['stage']} — {state.get('evaluations',0)} screening evaluations ({elapsed}s)"
                    except (OSError,ValueError,KeyError):
                        message=f'Target search starting ({elapsed}s)...'
                    on_progress(message)
                try:
                    code=process.wait(timeout=.2);break
                except subprocess.TimeoutExpired:
                    pass
            if code:
                log.flush()
                lines=[line for line in (folder/'search.log').read_text(errors='replace').splitlines() if line.strip()]
                reason=lines[-1][-400:] if lines else 'Worker exited without a diagnostic.'
                raise RuntimeError('Target search failed: '+reason+' See '+str(folder/'search.log'))
            result=validate_search_result(json.loads(output.read_text()),config,mode,target,vary_cavity)
            result['source']=str(output.resolve())
            if result['target_reached']:
                settings_file=folder/'optimized_settings.json'
                settings_file.write_text(json.dumps(result['best']['settings'],indent=2,allow_nan=False))
                result['settings_file']=str(settings_file.resolve())
            output.write_text(json.dumps(result,indent=2,allow_nan=False))
            return result
        finally:
            _stop(process)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=tuple(MODES),required=True);parser.add_argument('--target',type=float,required=True)
    parser.add_argument('--vary-cavity',action='store_true')
    args=parser.parse_args()
    config=validate(json.loads(Path(args.config).read_text()))
    result=worker(config,args.mode,args.target,Path(args.output).parent,vary_cavity=args.vary_cavity)
    Path(args.output).write_text(json.dumps(result,allow_nan=False))
