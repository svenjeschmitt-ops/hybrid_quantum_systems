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

from hybrid_settings import (DEFAULTS, calculated_swap_field, calculated_transmon_frequency,
                             derived, resonant_flux, resonant_field, validate)
from assistant_calculations import (MODES, CONTROL_FIELDS, ComparisonCancelled, _score,
                                    evaluate_backend)


class SearchBudget(Exception):
    pass


class StageReached(Exception):
    pass


def requested_target(question):
    """Explicit percentages only; ordinary resonance questions stay inexpensive."""
    if not re.search(r'fid\w*', question, re.I) or not re.search(r'\b(?:target|reach|achiev\w*|at\s+least|below|want|improv\w*|adjust|get)\b', question, re.I):
        return None
    match = re.search(r'(?:at\s+least|target|reach|achieve|want|below)\D{0,30}?(\d+(?:\.\d+)?)\s*(?:%|percent)', question, re.I)
    if not match:
        matches=list(re.finditer(r'(\d+(?:\.\d+)?)\s*(?:%|percent)',question,re.I))
        match=matches[-1] if matches else None
    if not match:
        return None
    value = float(match.group(1))/100
    if not 0 < value <= 1:
        raise ValueError('Target fidelity must be greater than 0 and at most 100%.')
    return value


def search_space(config, mode):
    """Peak detunings and log durations avoid ill-scaled flux/voltage coordinates."""
    import numpy as np
    c = validate(config)
    if mode not in MODES:
        raise ValueError('Unsupported search output mode.')
    if mode.startswith('Simultaneous cavity resonance:'):
        keys = ['SIM_HOLD_SCALE', 'OMEGA', 'GAP']
        bounds = [(math.log(.4), math.log(2.5)),
                  (math.log(c['OMEGA']*.5), math.log(c['OMEGA']*2)), (0, max(.03, c['GAP']*2))]
        if not (c['CONSTANT_FLUX'] and c['CONSTANT_FIELD']):
            keys.append('SIM_RAMP')
            bounds.append((math.log(c['SIM_RAMP']*.2), math.log(c['SIM_RAMP']*5)))
        seed=dict(c)
        if c['CONSTANT_FLUX']:
            try:
                seed['PHI_SWAP']=resonant_flux(c)
            except ValueError:
                pass
            keys.append('PHI_SWAP')
            bounds.append((max(0,seed['PHI_SWAP']-.025),min(.5,seed['PHI_SWAP']+.025)))
        if c['CONSTANT_FIELD']:
            try:
                seed['V_SWAP']=c['GAP_CM']*(resonant_field(c)-c['STRAY_FIELD'])
            except ValueError:
                pass
            keys.append('V_SWAP')
            width=max(.02,abs(seed['V_SWAP'])*.25)
            bounds.append((seed['V_SWAP']-width,seed['V_SWAP']+width))
        linear={'GAP','PHI_SWAP','V_SWAP'}
        def encode(values):
            x=np.array([values[key] if key in linear else math.log(values[key]) for key in keys])
            return np.clip(x,[a for a,b in bounds],[b for a,b in bounds])
        def decode(x):
            return validate({**c, **{key:float(value) if key in linear else math.exp(value) for key,value in zip(keys,x)}})
        return keys, bounds, encode(c), [encode(seed),encode(c)], decode
    stark = c['K_R1']-c['K_R0']
    if stark == 0:
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
    if tbound[0]>=tbound[1] or abound[0]>=abound[1]:
        raise ValueError('No reachable resonance-crossing interval exists within the search bounds.')
    keys = ['PHI_SWAP','TC_PULSE_SCALE','V_SWAP','AC_PULSE_SCALE']
    bounds = [tbound,(math.log(.4),math.log(25)),abound,(math.log(.4),math.log(25))]
    def encode(values):
        return np.array([calculated_transmon_frequency(values,values['PHI_SWAP'])-values['FC'],
                         math.log(values['TC_PULSE_SCALE']),
                         values['F_A_ZERO']+stark*calculated_swap_field(values)**2-values['FC'],
                         math.log(values['AC_PULSE_SCALE'])])
    def decode(x):
        phi = resonant_flux(dict(c,FC=c['FC']+float(x[0])))
        field = math.sqrt((c['FC']+float(x[2])-c['F_A_ZERO'])/stark)
        return validate(dict(c,PHI_SWAP=phi,TC_PULSE_SCALE=math.exp(x[1]),
                             V_SWAP=c['GAP_CM']*(field-c['STRAY_FIELD']),AC_PULSE_SCALE=math.exp(x[3])))
    def clip(x):
        return np.clip(x,[a for a,b in bounds],[b for a,b in bounds])
    old = derived(DEFAULTS)
    seed = np.array([old['transmon_swap_detuning_MHz'],
        math.log(DEFAULTS['TC_PULSE_SCALE']*math.sqrt(max(.1,abs(park_t)/max(1,abs(old['transmon_park_frequency_MHz']-DEFAULTS['FC']))))),
        old['atom_swap_detuning_MHz'],
        math.log(DEFAULTS['AC_PULSE_SCALE']*math.sqrt(max(.1,abs(park_a)/150)))])
    seeds = [clip(seed),clip(encode(c)),clip([old['transmon_swap_detuning_MHz'],math.log(c['TC_PULSE_SCALE']),
                                           old['atom_swap_detuning_MHz'],math.log(c['AC_PULSE_SCALE'])])]
    return keys,bounds,clip(encode(c)),seeds,decode


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


def screen(b, mode, stage='joint'):
    """Noiseless endpoint screening only; never used as final target evidence."""
    import numpy as np
    import qutip as qt
    options=dict(store_states=True,atol=1e-11,rtol=1e-9,method='dop853',max_step=b.MAX_STEP,nsteps=100000)
    if mode.startswith('Simultaneous cavity resonance:'):
        from simultaneous_coupling import model
        scores=[]
        for direction in MODES[mode]:
            h,_,info,*_=model(b,False,direction)
            times=np.unique([0.,*[x for pair in info['stages'].values() for x in pair]])
            initial=qt.tensor(b.e,qt.basis(b.NC,0),b.h0) if direction=='write' else b.retrieval_initial(0,1)
            target=qt.tensor(b.g,qt.basis(b.NC,0),b.h1) if direction=='write' else qt.tensor(b.e,qt.basis(b.NC,0),b.h0)
            final=qt.sesolve(h,initial,times,options=options).states[-1]
            scores.append(float(qt.expect(target.proj(),final)))
        return min(scores)
    scores=[]
    if stage in ('tc','ac') or 'write' in MODES[mode]:
        end=b.TC[1] if stage=='tc' else b.MAP[1]
        times=np.unique([0.,*[x for pair in (b.PREP,b.TC,b.AC,b.MAP) for x in pair if x<=end]])
        initial=qt.tensor(b.e,qt.basis(b.NC,0),b.h0)
        final=qt.sesolve(b.H,initial,times,options=options).states[-1]
        target=qt.tensor(b.g,qt.basis(b.NC,1),b.r0) if stage=='tc' else qt.tensor(b.g,qt.basis(b.NC,0),b.h1)
        scores.append(float(qt.expect(target.proj(),final)))
    if stage=='joint' and 'read' in MODES[mode]:
        h,_,stages,*_=b.retrieval_model()
        times=np.unique([0.,*[x for _,pair in stages for x in pair]])
        final=qt.sesolve(h,b.retrieval_initial(0,1),times,options=options).states[-1]
        scores.append(float(qt.expect(qt.tensor(b.e,qt.basis(b.NC,0),b.h0).proj(),final)))
    return min(scores)


def optimize_controls(config, mode, target, directory, max_evaluations=320, evaluator=None):
    import numpy as np
    from scipy.optimize import minimize, differential_evolution
    keys,bounds,current,seeds,decode=search_space(config,mode)
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
        for seed in seeds:
            evaluate(seed,'joint')
        if not mode.startswith('Simultaneous cavity resonance:') and best_score<goal:
            seed=seeds[0]
            tc=local(seed,[0,1],'tc',.985,80)
            ac=local(tc,[2,3],'ac',max(.95,goal),90)
            evaluate(ac,'joint')
            if best_score<goal:
                other=local(best_x.copy(),[0,1,2,3],'joint',goal,100)
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


def validate_search_result(result, config, mode, target):
    c=validate(config)
    if result['mode']!=mode or result['target']!=target or result['baseline']['settings']!=c:
        raise ValueError('Search evidence does not match the captured request.')
    best=result['best']
    if validate(best['settings'])!=best['settings']:
        raise ValueError('Search candidate settings are not canonical.')
    altered={k for k in c if best['settings'][k]!=c[k]}
    allowed=set(search_space(c,mode)[0])
    if altered-allowed:
        raise ValueError('Search changed fixed scientific assumptions.')
    before,after=_score(result['baseline'],mode),_score(best,mode)
    if best['mode']!=mode:
        raise ValueError('Search candidate mode mismatch.')
    result=dict(result,baseline_score=before,best_score=after,target_reached=after>=target,
                changes={k:best['settings'][k] for k in sorted(altered)})
    return result


def search_answer(result, config, mode, target):
    result=validate_search_result(result,config,mode,target)
    lines=[f"Target: **{target:.2%}** minimum joint fidelity for logical 0, 1 and + across the selected directions.",
           f"Validated score: **{result['baseline_score']:.4%} → {result['best_score']:.4%}**.",
           'Storage, noise, cavity frequency, basis and tolerance bounds were held fixed.','',
           '| Parameter | Current | Found |','|---|---:|---:|']
    for key in result.get('searched_parameters',[]):
        lines.append(f"| {key} | {config[key]} | {result['best']['settings'][key]} |")
    lines+=['','| Direction | Logical 0 | Logical 1 | Logical + |','|---|---:|---:|---:|']
    for direction,data in result['best']['directions'].items():
        lines.append(f"| {direction} | "+' | '.join(f'{v:.4%}' for v in data['sampled_logical_lows'])+' |')
    if result['target_reached']:
        lines+=['','**Target reached in the full noisy simulation**, including storage and configured tolerance samples.']
    else:
        lines+=['','**Target not reached within this search budget.** The best tested result remains below the request; no target-reaching proposal is offered.']
    lines+=['','Screening used noiseless endpoint populations; the reported fidelities are from full noisy validation. '
            'This is a bounded search, not a global optimum or a guarantee for all qubit inputs or unsampled errors.',
            'Numerical evidence: '+result.get('source','local search')]
    if result['target_reached'] and result['changes']:
        lines+=['','```panel-settings',json.dumps(dict(changes=result['changes'],evidence='measured simulation result',
            reason=f"Multi-parameter search validated minimum fidelity {result['best_score']:.6%}, meeting target {target:.2%}.")),'```']
    return '\n'.join(lines)


def worker(config, mode, target, directory, max_evaluations=320):
    directory=Path(directory)
    def validate_point(c,label):
        folder=directory/label;folder.mkdir(exist_ok=True)
        state=directory/'search_state.json'
        state.write_text(json.dumps(dict(stage='Full noisy validation: '+label,evaluations=0)))
        b=load_backend(c,directory)
        return evaluate_backend(b,mode,folder)
    baseline=validate_point(config,'baseline')
    best=baseline
    screening=dict(parameters=search_space(config,mode)[0],evaluations=0,candidates=[])
    if _score(baseline,mode)<target:
        screening=optimize_controls(config,mode,target,directory,max_evaluations)
        for index,candidate in enumerate(screening['candidates']):
            point=validate_point(candidate['settings'],f'candidate_{index}')
            if _score(point,mode)>_score(best,mode):
                best=point
            if _score(best,mode)>=target:
                break
    result=dict(mode=mode,target=target,baseline=baseline,best=best,
                searched_parameters=screening['parameters'],screening_evaluations=screening['evaluations'])
    return validate_search_result(result,config,mode,target)


def run_target_search(config, mode, target, cancelled, project_root, on_progress=None, timeout=2400):
    from codex_assistant import _stop
    config=validate(config)
    if not 0<target<=1:
        raise ValueError('Target fidelity must be greater than 0 and at most 100%.')
    search_space(config,mode)
    if cancelled.is_set():
        raise ComparisonCancelled()
    root=Path(project_root)/'output'/'assistant_searches';root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='search-',dir=root))
    conf=folder/'config.json';conf.write_text(json.dumps(config))
    output=folder/'result.json'
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(conf),
             '--output',str(output),'--mode',mode,'--target',str(target)]
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
                raise RuntimeError('Target search failed. See '+str(folder/'search.log'))
            result=validate_search_result(json.loads(output.read_text()),config,mode,target)
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
    args=parser.parse_args()
    config=validate(json.loads(Path(args.config).read_text()))
    result=worker(config,args.mode,args.target,Path(args.output).parent)
    Path(args.output).write_text(json.dumps(result,allow_nan=False))
