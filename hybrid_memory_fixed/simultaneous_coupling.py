"""Simultaneous resonant cavity-mediated write, separate from sequential pulses.

Frequencies are MHz, angular couplings rad/us, times us. The full cavity
remains explicit: no direct transmon-atom exchange term is introduced.
"""
import numpy as np
import simulation_progress as progress
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


def model(b, include_storage=True):
    """Shared exchange pulse with independently optional constant controls."""
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
    finish = mapping[1]+(b.STORAGE if include_storage else 0.0)

    def flux(t):
        if constant_flux:
            return info['interaction_flux']
        return b.PHI_PARK+(info['interaction_flux']-b.PHI_PARK)*plateau(t, start, hold, ramp)

    def field(t):
        if constant_field:
            return info['interaction_field_V_per_cm']
        return b.E_PARK+(info['interaction_field_V_per_cm']-b.E_PARK)*plateau(t, start, hold, ramp)

    def voltage(t):
        return c['V_SWAP'] if constant_field else b.GAP_CM*(field(t)-b.STRAY_FIELD)

    def dt(t):
        return b.TWOPI*(b.transmon_frequency(flux(t))-b.FC)

    def d0(t):
        return b.TWOPI*b.K_R0*field(t)**2

    def d1(t):
        return b.TWOPI*(b.F_A_ZERO-b.FC+b.K_R1*field(t)**2)

    def dw(t):
        return b.TWOPI*(b.F_WRONG_SPLIT+b.K_WRONG*field(t)**2)

    # Optical carriers follow the field used during preparation and mapping.
    optical_r0 = d0(0)
    optical_r1 = d1(0)

    def prep_drive(t):
        return .5*b.optical_envelope(t, *prep)*np.exp(-1j*optical_r0*(t-prep[0]))

    def map0(t):
        return .5*b.optical_envelope(t, *mapping)*np.exp(1j*optical_r0*(t-mapping[0]))

    def map1(t):
        return .5*b.optical_envelope(t, *mapping)*np.exp(1j*optical_r1*(t-mapping[0]))

    gt, ga = info['g_TC_rad_per_us'], info['g_AC_rad_per_us']
    interaction = gt*(b.a.dag()*b.sm+b.a*b.sm.dag()) + ga*(b.a*b.Rplus+b.a.dag()*b.Rplus.dag())
    terms = [interaction + b.H_anharm, [b.Nt, dt], [b.Pr0, d0], [b.Pr1, d1], [b.Pwrong_base, dw],
             *b.extra_atomic_terms(field)]
    for op, coeff in ((b.Bprep, prep_drive), (b.Bmap0, map0), (b.Bmap1, map1)):
        terms.extend([[op, coeff], [op.dag(), lambda t, f=coeff: np.conj(f(t))]])
    h = qt.QobjEvo(terms)
    points = [np.linspace(0, mapping[1], 1401), np.array([
        *prep, start, start+ramp, end-ramp, end, *mapping])]
    if include_storage and b.STORAGE:
        points.append(np.linspace(mapping[1], finish, 301))
    info.update(hold_us=hold, ramp_us=ramp, constant_flux=constant_flux,
                constant_field=constant_field, continuous_resonance=constant_flux and constant_field and info['on_resonance'], mapping_end_us=mapping[1],
                stages={'Preparation': prep, 'Simultaneous exchange': (start, end),
                        'Optical mapping': mapping, 'Storage': (mapping[1], finish)})
    return h, np.unique(np.concatenate(points)), info, flux, voltage, field


def simulate(b, alpha, beta, noisy=True, include_storage=True, storage_solver=None):
    h, times, info, *_ = model(b, include_storage)
    initial = qt.tensor((alpha*b.g+beta*b.e).unit(), qt.basis(b.NC, 0), b.h0)
    options = dict(store_states=True, atol=1e-11, rtol=1e-9, method='dop853',
                   max_step=min(b.MAX_STEP, info['ramp_us']/4) if info['ramp_us'] else b.MAX_STEP, nsteps=100000)
    dynamics_times = times[times <= info['mapping_end_us']]
    if noisy:
        result = qt.mesolve(h, qt.ket2dm(initial), dynamics_times, c_ops=b.C_OPS, options=options)
    else:
        result = qt.sesolve(h, initial, dynamics_times, options=options)
    states = list(result.states)
    storage_times = times[times > info['mapping_end_us']]-info['mapping_end_us']
    if len(storage_times):
        storage_hamiltonian = h(info['mapping_end_us'])
        if noisy:
            solver = storage_solver or b.ParkedSolver(storage_hamiltonian, b.C_OPS)
            held = solver.run(states[-1], np.r_[0., storage_times]).states
        else:
            held = qt.sesolve(storage_hamiltonian, states[-1], np.r_[0., storage_times],
                             options=options).states
        states.extend(held[1:])
    return times, states, info


def run(b, save=False):
    """Panel output: populations, controls, joint fidelity and logical inputs."""
    progress.update(0, "Calibrating simultaneous resonance")
    _, calibration, _ = simulate(b, 1, 1, noisy=False, include_storage=False)
    progress.update(.15, "Calculating simultaneous dynamics and storage")
    phase = float(np.angle(calibration[-1].ptrace(2)[0, 1]))
    rotation = b.memory_rotation(phase)
    theta, phi = np.deg2rad([b._SETTINGS['INPUT_THETA'], b._SETTINGS['INPUT_PHI']])
    alpha, beta = np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)
    h, _, setup, *_ = model(b)
    storage_solver = (b.ParkedSolver(h(setup['mapping_end_us']), b.C_OPS) if b.STORAGE else None)
    times, states, info = simulate(b, alpha, beta, storage_solver=storage_solver)
    target = qt.tensor(b.g, qt.basis(b.NC, 0), rotation.dag()*(alpha*b.h0+beta*b.h1))
    fidelity = np.real(qt.expect(target.proj(), states))
    operators = (b.Pe, b.n, b.Ph0, b.Ph1, b.Pr0, b.Pr1, b.Plost, b.Pwrong)
    names = ('transmon_excited', 'cavity_photons', 'h0', 'h1', 'r0', 'r1', 'loss', 'wrong')
    pops = dict(zip(names, np.real(qt.expect(operators, states))))
    basis = []
    output_states = [qt.tensor(b.g, qt.basis(b.NC, 0), atom) for atom in (b.h0, b.h1)]
    progress.update(.40, "Calculating simultaneous logical inputs")
    for label, x, y in (('g to h0', 1, 0), ('e to h1', 0, 1), ('+ to +', 1/np.sqrt(2), 1/np.sqrt(2))):
        progress.update(.40+.5*len(basis)/3, f'Simultaneous input {len(basis)+1}/3')
        print(f'Calculating simultaneous-resonance input {label}…', flush=True)
        _, trajectory, _ = simulate(b, x, y, storage_solver=storage_solver)
        goal = qt.tensor(b.g, qt.basis(b.NC, 0), rotation.dag()*(x*b.h0+y*b.h1))
        final = trajectory[-1]
        logical_probs = [float(np.real(qt.expect(output.proj(), final))) for output in output_states]
        wrong_prob = float(np.real(qt.expect(b.Pwrong, final)))
        basis.append({'input': label,
                      'joint_fidelity_after_storage': float(qt.expect(goal.proj(), final)),
                      'joint_output_probabilities': [*logical_probs, wrong_prob,
                                                     1-sum(logical_probs)-wrong_prob]})
    index = int(np.argmin(abs(times-info['mapping_end_us'])))
    summary = dict(joint_fidelity_after_mapping=float(fidelity[index]),
                   joint_fidelity_after_storage=float(fidelity[-1]), calibrated_phase_rad=phase,
                   **{k: v for k, v in info.items() if k != 'stages'})
    progress.update(.90, 'Building simultaneous resonance figure')
    print(f'Simultaneous resonance (nominal): constant flux={info["constant_flux"]}; constant field={info["constant_field"]}.', flush=True)
    print(f"Interaction flux={info['interaction_flux']:.9g}; voltage={info['interaction_voltage_V']:.9g} V", flush=True)
    print(f"g_TC/2pi={info['g_TC_rad_per_us']/b.TWOPI:.6g} MHz; g_AC/2pi={info['g_AC_rad_per_us']/b.TWOPI:.6g} MHz", flush=True)
    print(f"Hold={info['hold_us']:.6g} us; ideal resonant transfer reference={info['ideal_transfer_limit']:.6f}", flush=True)
    print(f'Joint fidelity after mapping={fidelity[index]:.6f}; after storage={fidelity[-1]:.6f}', flush=True)
    _, _, _, flux, voltage, field = model(b)
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
    population_axis.set(ylabel='Population / photon number', title='Write transfer dynamics')
    b.plot_resolved_populations(population_axis, times, states, transfer)
    b.population_legend(population_axis)
    held_coherence = float(2*abs(states[-1].ptrace(2)[0, 1]))
    rows = [
        ['Total write time (preparation through mapping)', f"{info['mapping_end_us']*1000:.2f} ns"],
        ['Storage hold after writing', f'{b.STORAGE:g} µs'],
        ['Total write + storage duration', f"{info['mapping_end_us']+b.STORAGE:.6g} µs"],
        ['Exchange hold', f"{info['hold_us']*1000:.2f} ns"],
        ['Flux control', 'Constant entered value' if info['constant_flux'] else 'Pulsed from park'],
        ['Electric field control', 'Constant entered value' if info['constant_field'] else 'Pulsed from park'],
        ['Ideal resonant reference (initial e,0,r0)', f"{info['ideal_transfer_limit']:.2%}"],
        ['Joint fidelity: g → h0', f"{basis[0]['joint_fidelity_after_storage']:.2%}"],
        ['Joint fidelity: e → h1', f"{basis[1]['joint_fidelity_after_storage']:.2%}"],
        ['Joint fidelity: + → +', f"{basis[2]['joint_fidelity_after_storage']:.2%}"],
        ['Selected input: fidelity before storage', f'{fidelity[index]:.2%}'],
        ['Selected input: fidelity after storage', f'{fidelity[-1]:.2%}'],
        ['Selected input: atomic coherence 2|ρ(h0,h1)|', f'{held_coherence:.6f}'],
        ['Selected input: residual transmon e |1>', f"{pops['transmon_excited'][-1]:.3e}"],
        ['Selected input: residual mean cavity photons', f"{pops['cavity_photons'][-1]:.3e}"],
        ['Selected input: residual r0 + r1 population', f"{pops['r0'][-1]+pops['r1'][-1]:.3e}"],
        ['Selected input: atomic loss population', f"{pops['loss'][-1]:.3e}"],
        ['Selected input: wrong-Rydberg population', f"{pops['wrong'][-1]:.3e}"],
    ]
    performance_axis.set_title('Write + storage summary (nominal)')
    table = b.results_table(performance_axis, ['Metric', 'Nominal'], rows,
                            widths=[.76, .24], bbox=(0, .04, 1, .94), fontsize=7.5)
    for row in range(1, len(rows)+1):
        table[row, 0].set_text_props(ha='left')
    labels = ['g → h0', 'e → h1', '+ → +']
    probabilities = np.asarray([row['joint_output_probabilities'] for row in basis])
    b.plot_basis_columns(basis_axis, labels, probabilities, 'Write outputs after storage')
    basis_axis.set_box_aspect((4, 4, 3), zoom=.98)
    basis_axis.xaxis.labelpad = -5
    fig.supxlabel(f'Selected input: θ={np.rad2deg(theta):g}°, φ={np.rad2deg(phi):g}°.\n'
                  f'Initial transmon populations: g = {100*abs(alpha)**2:.2f}%, '
                  f'e = {100*abs(beta)**2:.2f}%.', fontsize=9)
    for ax in (control_axis, detuning_axis, population_axis):
        ax.set_xlabel('Time (us)')
        for name, (start, end) in info['stages'].items():
            if name != 'Storage':
                ax.axvline(start, color='gray', alpha=.15)
                ax.axvline(end, color='gray', alpha=.15)
    fig.suptitle('Simultaneous cavity exchange: transmon to atom (nominal)'
                 + ('' if info['on_resonance'] else ' — detuned entered controls'))
    fig.canvas.manager.set_window_title('Simultaneous cavity resonance')
    summary.update(displayed_metrics=rows, selected_coherence_after_storage=held_coherence,
                   populations_after_storage={name: float(values[-1]) for name, values in pops.items()})
    if b._ASSISTANT_OUTPUT:
        write_report(b._ASSISTANT_OUTPUT, 'write', b._SETTINGS,
                     dict(protocol='simultaneous_cavity_resonance', continuous_resonance=info['continuous_resonance'],
                          constant_flux=info['constant_flux'], constant_field=info['constant_field'], noise_enabled=True,
                          tolerance_scan=False, input_theta_degrees=b._SETTINGS['INPUT_THETA'],
                          input_phi_degrees=b._SETTINGS['INPUT_PHI'],
                          joint_output_labels=['g,0,h0', 'g,0,h1', 'wrong_rydberg', 'remaining_other'],
                          basis_evaluation='After writing AND storage; original transmon g/e/+ inputs.',
                          stages_us=info['stages']), summary, traces, basis)
    if save:
        from pathlib import Path
        import json
        folder = Path(b.__file__).resolve().parent
        fig.savefig(folder/'hybrid_simultaneous.png', dpi=180)
        (folder/'hybrid_simultaneous.json').write_text(json.dumps(
            dict(summary=summary, traces=traces, basis_outputs=basis), allow_nan=False), encoding='utf-8')
    return summary
