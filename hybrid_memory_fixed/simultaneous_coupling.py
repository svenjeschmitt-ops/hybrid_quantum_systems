"""Bidirectional simultaneous resonant cavity transfer, separate from sequential pulses.

Frequencies are MHz, angular couplings rad/us, times us. The full cavity
remains explicit: no direct transmon-atom exchange term is introduced.
"""
import numpy as np
import simulation_progress as progress
import simulation_stop as stopping
import tolerance_bands
import qutip as qt

from hybrid_settings import validate, coupling_strengths, resonant_flux, resonant_field, calculated_swap_field, calculated_transmon_frequency
from simulation_output import trace, write_report


def resonance_settings(settings):
    """Use entered constant controls; auto-tune only controls that remain pulsed."""
    c = validate(settings)
    try:
        phi = c["PHI_SWAP"] if c["CONSTANT_FLUX"] else resonant_flux(c)
        field = calculated_swap_field(c) if c["CONSTANT_FIELD"] else resonant_field(c)
    except ValueError as exc:
        raise ValueError(f'Simultaneous resonance: {exc}') from exc
    # Evaluate the coupling at the actual interaction flux, including detuned constants.
    rates = coupling_strengths(dict(c, PHI_SWAP=phi))
    gt, ga = [2*np.pi*rates[key] for key in ('transmon_coupling_MHz', 'atom_coupling_MHz')]
    dt = calculated_transmon_frequency(c, phi)-c['FC']
    da = c['F_A_ZERO']+(c['K_R1']-c['K_R0'])*field**2-c['FC']
    return dict(interaction_flux=phi, interaction_field_V_per_cm=field,
                transmon_detuning_MHz=dt, atom_detuning_MHz=da,
                on_resonance=bool(abs(dt)<1e-8 and abs(da)<1e-8),
                interaction_voltage_V=c['GAP_CM']*(field-c['STRAY_FIELD']),
                g_TC_rad_per_us=gt, g_AC_rad_per_us=ga,
                reference_hold_us=np.pi/np.hypot(gt, ga),
                ideal_transfer_limit=4*(gt*ga/(gt*gt+ga*ga))**2)


def plateau(t, start, hold, ramp):
    """A flat resonant interval with smooth ramps from and back to park."""
    u = t-start
    if u <= 0 or u >= 2*ramp+hold:
        return 0.0
    if u < ramp:
        return float(np.sin(np.pi*u/(2*ramp))**2)
    if u <= ramp+hold:
        return 1.0
    return float(np.sin(np.pi*(2*ramp+hold-u)/(2*ramp))**2)


def model(b, include_storage=True, direction="write"):
    """Shared exchange pulse with independently optional constant controls."""
    if direction not in ("write", "read"):
        raise ValueError("direction must be write or read")
    c = b._SETTINGS
    info = resonance_settings(c)
    constant_flux = bool(c['CONSTANT_FLUX'])
    constant_field = bool(c['CONSTANT_FIELD'])
    ramp = 0.0 if constant_flux and constant_field else c['SIM_RAMP']
    hold = c['SIM_HOLD_SCALE']*info['reference_hold_us']
    prep = (0.0, b.T_OPT)
    start = prep[1]+b.GAP
    end = start+2*ramp+hold
    mapping = (end+b.GAP, end+b.GAP+b.T_OPT)
    finish = mapping[1]+(b.STORAGE if include_storage and direction == "write" else 0.0)

    def flux(t):
        if constant_flux:
            return info['interaction_flux']+b.TOLERANCE_OFFSETS['FLUX']
        return b.PHI_PARK+(info['interaction_flux']-b.PHI_PARK)*plateau(t, start, hold, ramp)+b.TOLERANCE_OFFSETS['FLUX']

    def field(t):
        if constant_field:
            return info['interaction_field_V_per_cm']+b.TOLERANCE_OFFSETS['EFIELD']
        return b.E_PARK+(info['interaction_field_V_per_cm']-b.E_PARK)*plateau(t, start, hold, ramp)+b.TOLERANCE_OFFSETS['EFIELD']

    def voltage(t):
        return c['V_SWAP'] if constant_field else b.GAP_CM*(field(t)-b.TOLERANCE_OFFSETS['EFIELD']-b.STRAY_FIELD)

    def dt(t):
        return b.TWOPI*(b.transmon_frequency(flux(t))+b.TOLERANCE_OFFSETS['FT']-b.FC)

    def d0(t):
        return b.TWOPI*b.K_R0*field(t)**2

    def d1(t):
        return b.TWOPI*(b.F_A_ZERO+b.TOLERANCE_OFFSETS['FA']-b.FC+b.K_R1*field(t)**2)

    def dw(t):
        return b.TWOPI*(b.F_WRONG_SPLIT+b.K_WRONG*field(t)**2)

    # Fix the laser frequencies to the nominal controls, including in error scenarios.
    nominal_field = info['interaction_field_V_per_cm'] if constant_field else b.E_PARK
    optical_r0 = b.TWOPI*b.K_R0*nominal_field**2
    optical_r1 = b.TWOPI*(b.F_A_ZERO-b.FC+b.K_R1*nominal_field**2)

    def prep_drive(t):
        return .5*b.optical_envelope(t, *prep)*np.exp(-1j*optical_r0*(t-prep[0]))

    def map0(t):
        return .5*b.optical_envelope(t, *mapping)*np.exp(1j*optical_r0*(t-mapping[0]))

    def map1(t):
        return .5*b.optical_envelope(t, *mapping)*np.exp(1j*optical_r1*(t-mapping[0]))

    gt, ga = info['g_TC_rad_per_us'], info['g_AC_rad_per_us']
    interaction = gt*(b.a.dag()*b.sm+b.a*b.sm.dag()) + ga*(b.a*b.Rplus+b.a.dag()*b.Rplus.dag())
    terms = [interaction + b.H_anharm, [b.n, b.cavity_error], [b.Nt, dt], [b.Pr0, d0], [b.Pr1, d1], [b.Pwrong_base, dw],
             *b.extra_atomic_terms(field)]
    drives = ((b.Bprep, prep_drive), (b.Bmap0, map0), (b.Bmap1, map1))
    if direction == "read":
        # Excite both memory arms before exchange; de-excite r0 afterwards.
        def unmap0(t):
            return .5*b.optical_envelope(t, *prep)*np.exp(-1j*optical_r0*(t-prep[0]))
        def unmap1(t):
            return .5*b.optical_envelope(t, *prep)*np.exp(-1j*optical_r1*(t-prep[0]))
        drives = ((b.Bmap0.dag(), unmap0), (b.Bmap1.dag(), unmap1),
                  (b.Bprep.dag(), map0))
    for op, coeff in drives:
        terms.extend([[op, coeff], [op.dag(), lambda t, f=coeff: np.conj(f(t))]])
    h = qt.QobjEvo(terms)
    points = [np.linspace(0, mapping[1], 1401), np.array([
        *prep, start, start+ramp, end-ramp, end, *mapping])]
    if include_storage and b.STORAGE and direction == "write":
        points.append(np.linspace(mapping[1], finish, 301))
    info.update(hold_us=hold, ramp_us=ramp, constant_flux=constant_flux,
                constant_field=constant_field, continuous_resonance=constant_flux and constant_field and info['on_resonance'], mapping_end_us=mapping[1],
                stages={'Preparation': prep, 'Simultaneous exchange': (start, end),
                        'Optical mapping': mapping, 'Storage': (mapping[1], finish)})
    info['direction'] = direction
    if direction == "read":
        info['stages'] = {'Memory to Rydberg': prep, 'Simultaneous exchange': (start, end),
                          'Return atom to h0': mapping}
        info['storage_before_read_us'] = b.STORAGE if include_storage else 0.0
    return h, np.unique(np.concatenate(points)), info, flux, voltage, field


def simulate(b, alpha, beta, noisy=True, include_storage=True, storage_solver=None, direction="write"):
    h, times, info, *_ = model(b, include_storage, direction)
    if direction == "write":
        initial = qt.tensor((alpha*b.g+beta*b.e).unit(), qt.basis(b.NC, 0), b.h0)
    else:
        initial = qt.tensor(b.g, qt.basis(b.NC, 0), (alpha*b.h0+beta*b.h1).unit())
        if include_storage and b.STORAGE:
            # Independent atomic input: storage precedes retrieval with no reset.
            if noisy:
                solver = storage_solver or b.ParkedSolver(h(0), b.C_OPS)
                initial = solver.run(initial, [0, b.STORAGE]).states[-1]
            else:
                initial = (-1j*h(0)*b.STORAGE).expm()*initial
    options = dict(store_states=True, atol=1e-11, rtol=1e-9, method='dop853',
                   max_step=min(b.MAX_STEP, info['ramp_us']/4) if info['ramp_us'] else b.MAX_STEP, nsteps=100000)
    dynamics_times = times[times <= info['mapping_end_us']]
    if noisy:
        result = stopping.solve(qt.mesolve, h, initial.proj() if initial.isket else initial, dynamics_times, c_ops=b.C_OPS, options=options)
    else:
        result = stopping.solve(qt.sesolve, h, initial, dynamics_times, options=options)
    states = list(result.states)
    storage_times = times[times > info['mapping_end_us']]-info['mapping_end_us']
    if len(storage_times):
        storage_hamiltonian = h(info['mapping_end_us'])
        if noisy:
            solver = storage_solver or b.ParkedSolver(storage_hamiltonian, b.C_OPS)
            held = solver.run(states[-1], np.r_[0., storage_times]).states
        else:
            held = stopping.solve(qt.sesolve, storage_hamiltonian, states[-1], np.r_[0., storage_times],
                             options=options).states
        states.extend(held[1:])
    return times, states, info


def control_samples(b, times, direction):
    _, _, _, flux, _, field = model(b, direction=direction)
    phi = np.asarray([flux(t) for t in times])
    electric = np.asarray([field(t) for t in times])
    offsets = b.TOLERANCE_OFFSETS
    return dict(flux=phi, field=electric,
                transmon_detuning=b.transmon_frequency(phi)+offsets['FT']-b.FC-offsets['FC'],
                atom_detuning=b.F_A_ZERO+offsets['FA']+(b.K_R1-b.K_R0)*electric**2-b.FC-offsets['FC'])


def run(b, save=False, direction="write"):
    """Panel output: populations, controls, joint fidelity and logical inputs."""
    writing = direction == "write"
    work = progress.transfer_stages(b._SETTINGS)
    title_direction = "transmon to atom" if writing else "atom to transmon"
    progress.update(0, "Calibrating simultaneous " + title_direction)
    _, calibration, _ = simulate(b, 1, 1, noisy=False, include_storage=not writing, direction=direction)
    progress.update(work["calibration"], "Calculating simultaneous dynamics and storage")
    phase = float(np.angle(calibration[-1].ptrace(2 if writing else 0)[0, 1]))
    rotation = b.memory_rotation(phase) if writing else b.IT+(np.exp(1j*phase)-1)*b.e.proj()
    def joint_target(x, y):
        if writing:
            return qt.tensor(b.g, qt.basis(b.NC, 0), rotation.dag()*(x*b.h0+y*b.h1))
        return qt.tensor(rotation.dag()*(x*b.g+y*b.e), qt.basis(b.NC, 0), b.h0)
    theta, phi = np.deg2rad([b._SETTINGS['INPUT_THETA'], b._SETTINGS['INPUT_PHI']])
    alpha, beta = np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)
    h, _, setup, *_ = model(b, direction=direction)
    storage_solver = (b.ParkedSolver(h(setup['mapping_end_us'] if writing else 0), b.C_OPS) if b.STORAGE else None)
    times, states, info = simulate(b, alpha, beta, storage_solver=storage_solver, direction=direction)
    target = joint_target(alpha, beta)
    fidelity = np.real(qt.expect(target.proj(), states))
    operators = (b.Pe, b.n, b.Ph0, b.Ph1, b.Pr0, b.Pr1, b.Plost, b.Pwrong)
    names = ('transmon_excited', 'cavity_photons', 'h0', 'h1', 'r0', 'r1', 'loss', 'wrong')
    pops = dict(zip(names, np.real(qt.expect(operators, states))))
    basis = []
    output_states = ([qt.tensor(b.g, qt.basis(b.NC, 0), atom) for atom in (b.h0, b.h1)] if writing else
                     [qt.tensor(transmon, qt.basis(b.NC, 0), b.h0) for transmon in (b.g, b.e)])
    input_labels = ('g to h0', 'e to h1', '+ to +') if writing else ('h0 to g', 'h1 to e', '+ to +')
    progress.update(work["storage"], "Calculating simultaneous logical inputs")
    for label, (x, y) in zip(input_labels, ((1, 0), (0, 1), (1/np.sqrt(2), 1/np.sqrt(2)))):
        progress.update(work['storage']+(work['basis']-work['storage'])*len(basis)/3, f'Simultaneous input {len(basis)+1}/3')
        print(f'Calculating simultaneous-resonance input {label}…', flush=True)
        _, trajectory, _ = simulate(b, x, y, storage_solver=storage_solver, direction=direction)
        goal = joint_target(x, y)
        final = trajectory[-1]
        logical_probs = [float(np.real(qt.expect(output.proj(), final))) for output in output_states]
        wrong_prob = float(np.real(qt.expect(b.Pwrong, final)))
        basis.append({'input': label,
                      'joint_fidelity_final': float(qt.expect(goal.proj(), final)),
                      'joint_output_probabilities': [*logical_probs, wrong_prob,
                                                     1-sum(logical_probs)-wrong_prob]})
    if writing:
        for row in basis:
            row['joint_fidelity_after_storage'] = row['joint_fidelity_final']
    index = int(np.argmin(abs(times-info['mapping_end_us'])))
    def evaluate_errors():
        grid, scenario_states, _ = simulate(b, alpha, beta, direction=direction)
        values = np.real(qt.expect(target.proj(), scenario_states))
        logical = []
        for j, (x, y) in enumerate(((1, 0), (0, 1), (1/np.sqrt(2), 1/np.sqrt(2)))):
            with progress.scope(.25+.7*j/3, .25+.7*(j+1)/3, 'Tolerance logical input'):
                _, logical_states, _ = simulate(b, x, y, direction=direction)
            logical.append(float(np.real(qt.expect(joint_target(x, y).proj(), logical_states[-1]))))
        return grid, scenario_states, values, scenario_states[-1], float(values[-1]), logical

    with progress.scope(work['basis'], .95, 'Simultaneous '+direction+' tolerance scenarios'):
        bands = tolerance_bands.compute(
            b, b._SETTINGS, direction, alpha, beta, phase, times,
            np.asarray([pops[name] for name in names]), fidelity, fidelity[-1],
            nominal_state=states[-1], nominal_basis=[row['joint_fidelity_final'] for row in basis],
            evaluator=evaluate_errors, control_sampler=lambda grid: control_samples(b, grid, direction),
            before_index=index)
    summary = dict(joint_fidelity_after_mapping=float(fidelity[index]),
                   joint_fidelity_after_storage=float(fidelity[-1]), calibrated_phase_rad=phase,
                   **{k: v for k, v in info.items() if k != 'stages'})
    progress.update(.95, 'Building simultaneous resonance figure')
    print(f'Simultaneous resonance (nominal): constant flux={info["constant_flux"]}; constant field={info["constant_field"]}.', flush=True)
    print(f"Interaction flux={info['interaction_flux']:.9g}; voltage={info['interaction_voltage_V']:.9g} V", flush=True)
    print(f"g_TC/2pi={info['g_TC_rad_per_us']/b.TWOPI:.6g} MHz; g_AC/2pi={info['g_AC_rad_per_us']/b.TWOPI:.6g} MHz", flush=True)
    print(f"Hold={info['hold_us']:.6g} us; ideal resonant transfer reference={info['ideal_transfer_limit']:.6f}", flush=True)
    print(f'Joint fidelity after mapping={fidelity[index]:.6f}; after storage={fidelity[-1]:.6f}' if writing else f'Joint fidelity after storage and readout={fidelity[-1]:.6f}', flush=True)
    _, _, _, flux, voltage, field = model(b, direction=direction)
    summary['multilevel_dynamics'] = b.multilevel_summary(states)
    traces = {
        'multilevel_populations': trace(times, **b.multilevel_metrics(states)),
        'populations': trace(times, **pops),
        'joint_fidelity': trace(times, fidelity=fidelity),
        'controls': trace(times, flux=[flux(t) for t in times], voltage_V=[voltage(t) for t in times]),
        'detunings_MHz': trace(times,
            transmon=[b.transmon_frequency(flux(t))-b.FC for t in times],
            atom=[b.F_A_ZERO+(b.K_R1-b.K_R0)*field(t)**2-b.FC for t in times]),
    }
    traces.update(tolerance_bands.report_traces(bands))
    summary['tolerance_ranges'] = bands['summary'] if bands else None
    # Use the same three-row structure and shared renderers as sequential writing.
    fig = b.plt.figure(figsize=(12, 10), layout='constrained')
    grid = fig.add_gridspec(3, 2)
    control_axis = fig.add_subplot(grid[0, 0])
    voltage_axis = control_axis.twinx()
    detuning_axis = fig.add_subplot(grid[0, 1])
    population_axis = fig.add_subplot(grid[1, :])
    performance_axis = fig.add_subplot(grid[2, 0])
    basis_axis = fig.add_subplot(grid[2, 1], projection='3d')
    x = np.linspace(0, info['mapping_end_us'], 2501)
    flux_line, = control_axis.plot(x, [flux(t) for t in x],
                                   color='tab:blue', label='Flux (left axis)')
    voltage_line, = voltage_axis.plot(x, [voltage(t) for t in x],
                                     color='tab:orange', linestyle='--',
                                     label='Voltage (right axis)')
    control_axis.set_ylabel('Flux / flux quantum', color='tab:blue')
    voltage_axis.set_ylabel('Electrode voltage (V)', color='tab:orange')
    control_axis.tick_params(axis='y', colors='tab:blue')
    voltage_axis.tick_params(axis='y', colors='tab:orange')
    control_axis.spines['left'].set_color('tab:blue')
    voltage_axis.spines['right'].set_color('tab:orange')
    control_axis.set_title('Flux and voltage control')
    voltage_axis.legend(handles=[flux_line, voltage_line], loc='upper center',
                        bbox_to_anchor=(.5, .88), ncol=2, fontsize=8, framealpha=1)
    detuning_axis.plot(x, [b.transmon_frequency(flux(t))-b.FC for t in x],
                       color='tab:blue', label='Transmon')
    detuning_axis.plot(x, [b.F_A_ZERO+(b.K_R1-b.K_R0)*field(t)**2-b.FC for t in x],
                       color='tab:orange', linestyle='--', label='Atom')
    detuning_axis.axhline(0, color='gray', linewidth=.7, alpha=.5)
    detuning_axis.set(title='Detuning from cavity', ylabel='Detuning (MHz)')
    detuning_axis.legend(fontsize=8)
    transfer = times <= info['mapping_end_us']
    for name, label in zip(names, ('Transmon e |1>', 'Cavity photons', 'h0', 'h1',
                                  'r0', 'r1', 'Loss', 'Unwanted Rydberg total')):
        population_axis.plot(times[transfer], pops[name][transfer], label=label)
    population_axis.set(ylabel='Population / photon number', title='Write transfer dynamics' if writing else 'Read transfer dynamics')
    b.plot_resolved_populations(population_axis, times, states, transfer)
    if bands:
        visible_bands = dict(bands, times=times[transfer],
                             low={key: value[..., transfer] for key, value in bands['low'].items()},
                             high={key: value[..., transfer] for key, value in bands['high'].items()})
        tolerance_bands.draw(visible_bands, control_axis, detuning_axis, population_axis)
    b.population_legend(population_axis)
    held_coherence = float(2*abs(states[-1].ptrace(2 if writing else 0)[0, 1]))
    rows = [
        ['Total write time (preparation through mapping)', f"{info['mapping_end_us']*1000:.2f} ns"],
        ['Storage hold after writing', f'{b.STORAGE:g} µs'],
        ['Total write + storage duration', f"{info['mapping_end_us']+b.STORAGE:.6g} µs"],
        ['Exchange hold', f"{info['hold_us']*1000:.2f} ns"],
        ['Flux control', 'Constant entered value' if info['constant_flux'] else 'Pulsed from park'],
        ['Electric field control', 'Constant entered value' if info['constant_field'] else 'Pulsed from park'],
        ['Ideal resonant reference (initial e,0,r0)', f"{info['ideal_transfer_limit']:.2%}"],
        ['Joint fidelity: g → h0', f"{basis[0]['joint_fidelity_final']:.2%}"],
        ['Joint fidelity: e → h1', f"{basis[1]['joint_fidelity_final']:.2%}"],
        ['Joint fidelity: + → +', f"{basis[2]['joint_fidelity_final']:.2%}"],
        ['Selected input: fidelity before storage', f'{fidelity[index]:.2%}'],
        ['Selected input: fidelity after storage', f'{fidelity[-1]:.2%}'],
        ['Selected input: atomic coherence 2|ρ(h0,h1)|', f'{held_coherence:.6f}'],
        ['Selected input: residual transmon e |1>', f"{pops['transmon_excited'][-1]:.3e}"],
        ['Selected input: residual mean cavity photons', f"{pops['cavity_photons'][-1]:.3e}"],
        ['Selected input: residual r0 + r1 population', f"{pops['r0'][-1]+pops['r1'][-1]:.3e}"],
        ['Selected input: atomic loss population', f"{pops['loss'][-1]:.3e}"],
        ['Selected input: wrong-Rydberg population', f"{pops['wrong'][-1]:.3e}"],
    ]
    if not writing:
        rows[0][0] = 'Total read time (unmapping through atom reset)'
        rows[1][0] = 'Storage hold before retrieval'
        rows[2][0] = 'Total storage + read duration'
        rows[6][0] = 'Ideal resonant reference (initial g,0,r1)'
        rows[7][0], rows[8][0] = 'Joint fidelity: h0 → g', 'Joint fidelity: h1 → e'
        rows[10][0], rows[10][1] = 'Selected input: joint fidelity after readout', f'{fidelity[-1]:.2%}'
        rows[11][0], rows[11][1] = 'Selected input: atom reset h0 population', f"{pops['h0'][-1]:.6f}"
        rows[12][0] = 'Selected input: transmon coherence 2|ρ(g,e)|'
        rows[13][0] = 'Selected input: transmon e population'
    performance_axis.set_title(('Write + storage summary' if writing else 'Storage + read summary')
                               + ('' if bands else ' (nominal)'))
    headers, widths = ['Metric', 'Nominal'], [.76, .24]
    if bands:
        keys = ['basis_0', 'basis_1', 'basis_plus',
                'before_fidelity' if writing else 'final_fidelity',
                'final_fidelity' if writing else 'atom_h0', 'coherence', 'transmon_e',
                'cavity_mean_photons', 'rydberg_population', 'atom_loss', 'atom_wrong_rydberg']
        formats = ['.2%']*4 + (['.2%'] if writing else ['.6f']) + ['.6f'] + ['.3e']*5
        for row, key, fmt in zip(rows[7:], keys, formats):
            bounds = bands['summary']['metric_ranges'][key]
            row.append(f"{bounds['low']:{fmt}}–{bounds['high']:{fmt}}")
        for row in rows[:7]:
            row.append('—')
        headers, widths = ['Metric', 'Nominal', 'With tolerances'], [.60, .16, .24]
    table = b.results_table(performance_axis, headers, rows,
                            widths=widths, bbox=(0, .04, 1, .94), fontsize=7 if bands else 7.5)
    for row in range(1, len(rows)+1):
        table[row, 0].set_text_props(ha='left')
    labels = ['g → h0', 'e → h1', '+ → +'] if writing else ['h0 → g', 'h1 → e', '+ → +']
    probabilities = np.asarray([row['joint_output_probabilities'] for row in basis])
    b.plot_basis_columns(basis_axis, labels, probabilities, 'Write outputs after storage' if writing else 'Read outputs after storage + retrieval', mode=direction)
    basis_axis.set_box_aspect((4, 4, 3), zoom=.98)
    basis_axis.xaxis.labelpad = -5
    fig.supxlabel(f'Selected input: θ={np.rad2deg(theta):g}°, φ={np.rad2deg(phi):g}°.\n'
                  f'Initial {"transmon" if writing else "atom"} populations: {"g" if writing else "h0"} = {100*abs(alpha)**2:.2f}%, '
                  f'{"e" if writing else "h1"} = {100*abs(beta)**2:.2f}%.', fontsize=9)
    for ax in (control_axis, detuning_axis, population_axis):
        ax.set_xlabel('Time (us)')
        for name, (start, end) in info['stages'].items():
            if name != 'Storage':
                ax.axvline(start, color='gray', alpha=.15)
                ax.axvline(end, color='gray', alpha=.15)
    fig.suptitle(f'Simultaneous cavity exchange: {title_direction}'
                 + (' — nominal lines, sampled tolerance ranges' if bands else ' (nominal)')
                 + ('' if info['on_resonance'] else ' — detuned entered controls'))
    fig.canvas.manager.set_window_title('Simultaneous cavity resonance: '+title_direction)
    summary.update(displayed_metrics=rows, selected_coherence_after_storage=held_coherence,
                   populations_after_storage={name: float(values[-1]) for name, values in pops.items()})
    if not writing:
        summary['joint_fidelity_after_readout'] = summary.pop('joint_fidelity_after_mapping')
        summary.pop('joint_fidelity_after_storage')
        summary.pop('selected_coherence_after_storage')
        summary.pop('populations_after_storage')
        summary.update(selected_joint_fidelity=float(fidelity[-1]),
                       selected_transmon_coherence=held_coherence,
                       populations_after_read={name: float(values[-1]) for name, values in pops.items()})
    if b._ASSISTANT_OUTPUT:
        write_report(b._ASSISTANT_OUTPUT, direction, b._SETTINGS,
                     dict(protocol='simultaneous_cavity_resonance', continuous_resonance=info['continuous_resonance'],
                          constant_flux=info['constant_flux'], constant_field=info['constant_field'], noise_enabled=True,
                          tolerance_scan=bands is not None, input_theta_degrees=b._SETTINGS['INPUT_THETA'],
                          input_phi_degrees=b._SETTINGS['INPUT_PHI'],
                          joint_output_labels=(['g,0,h0', 'g,0,h1', 'wrong_rydberg', 'remaining_other'] if writing else ['g,0,h0', 'e,0,h0', 'wrong_rydberg', 'remaining_other']),
                          basis_evaluation=('After writing AND storage; original transmon g/e/+ inputs.' if writing else 'Independent atomic h0/h1/+ inputs; storage BEFORE retrieval.'),
                          phase_correction_rad=phase, direction=direction,
                          stages_us=info['stages']), summary, traces, basis)
    if save:
        from pathlib import Path
        import json
        folder = Path(b.__file__).resolve().parent
        stem = 'hybrid_simultaneous' if writing else 'hybrid_simultaneous_read'
        fig.savefig(folder/(stem+'.png'), dpi=180)
        (folder/(stem+'.json')).write_text(json.dumps(
            dict(summary=summary, traces=traces, basis_outputs=basis), allow_nan=False), encoding='utf-8')
    return summary
