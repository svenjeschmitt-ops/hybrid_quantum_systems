"""Simultaneous resonant cavity-mediated write, separate from sequential pulses.

Frequencies are MHz, angular couplings rad/us, times us. The full cavity
remains explicit: no direct transmon-atom exchange term is introduced.
"""
import numpy as np
import qutip as qt

from hybrid_settings import validate, coupling_strengths
from simulation_output import trace, write_report


def resonance_settings(settings):
    """Find reachable bare resonance controls without modifying saved inputs."""
    c = validate(settings)
    ej = (c['FC'] + c['EC'])**2 / (8*c['EC'])
    ratio = ej/c['EJ_MAX']
    if not c['ASYMMETRY'] <= ratio <= 1:
        raise ValueError('Simultaneous resonance: cavity frequency is outside the transmon tuning range.')
    cos2 = (ratio**2-c['ASYMMETRY']**2)/(1-c['ASYMMETRY']**2)
    phi = float(np.arccos(np.sqrt(np.clip(cos2, 0, 1)))/np.pi)
    stark = c['K_R1']-c['K_R0']
    if stark == 0:
        if not np.isclose(c['F_A_ZERO'], c['FC'], rtol=0, atol=1e-9):
            raise ValueError('Simultaneous resonance: equal Stark coefficients cannot tune the atom to the cavity.')
        field = c['E_PARK']
    else:
        field2 = (c['FC']-c['F_A_ZERO'])/stark
        if field2 < 0:
            raise ValueError('Simultaneous resonance: the atomic Stark model cannot reach the cavity frequency.')
        magnitude = float(np.sqrt(field2))
        field = min((magnitude, -magnitude), key=lambda x: abs(x-c['E_PARK']))
    # Use the charge matrix element at resonance, rather than the sequential peak.
    rates = coupling_strengths(dict(c, PHI_SWAP=phi))
    gt, ga = [2*np.pi*rates[key] for key in ('transmon_coupling_MHz', 'atom_coupling_MHz')]
    return dict(resonant_flux=phi, resonant_field_V_per_cm=field,
                resonant_voltage_V=c['GAP_CM']*(field-c['STRAY_FIELD']),
                g_TC_rad_per_us=gt, g_AC_rad_per_us=ga,
                reference_hold_us=np.pi/np.hypot(gt, ga),
                ideal_transfer_limit=4*(gt*ga/(gt*gt+ga*ga))**2)


def plateau(t, start, hold, ramp):
    """Smooth synchronous ramps with a genuinely flat resonant plateau."""
    u = t-start
    if u <= 0 or u >= 2*ramp+hold:
        return 0.0
    if u < ramp:
        return float(np.sin(np.pi*u/(2*ramp))**2)
    if u <= ramp+hold:
        return 1.0
    return float(np.sin(np.pi*(2*ramp+hold-u)/(2*ramp))**2)


def model(b, include_storage=True):
    """Build an independent Hamiltonian; sequential globals are never retimed."""
    c = b._SETTINGS
    info = resonance_settings(c)
    ramp = c['SIM_RAMP']
    hold = c['SIM_HOLD_SCALE']*info['reference_hold_us']
    prep = (0.0, b.T_OPT)
    start = prep[1]+b.GAP
    end = start+2*ramp+hold
    mapping = (end+b.GAP, end+b.GAP+b.T_OPT)
    finish = mapping[1]+(b.STORAGE if include_storage else 0.0)

    def weight(t):
        return plateau(t, start, hold, ramp)

    def flux(t):
        return b.PHI_PARK+(info['resonant_flux']-b.PHI_PARK)*weight(t)

    def field(t):
        return b.E_PARK+(info['resonant_field_V_per_cm']-b.E_PARK)*weight(t)

    def voltage(t):
        return b.GAP_CM*(field(t)-b.STRAY_FIELD)

    def dt(t):
        return b.TWOPI*(b.transmon_frequency(flux(t))-b.FC)

    def d0(t):
        return b.TWOPI*b.K_R0*field(t)**2

    def d1(t):
        return b.TWOPI*(b.F_A_ZERO-b.FC+b.K_R1*field(t)**2)

    def dw(t):
        return b.TWOPI*(b.F_WRONG_SPLIT+b.K_WRONG*field(t)**2)

    def prep_drive(t):
        return .5*b.optical_envelope(t, *prep)*np.exp(-1j*b.D_R0_PARK*(t-prep[0]))

    def map0(t):
        return .5*b.optical_envelope(t, *mapping)*np.exp(1j*b.D_R0_PARK*(t-mapping[0]))

    def map1(t):
        return .5*b.optical_envelope(t, *mapping)*np.exp(1j*b.D_R1_PARK*(t-mapping[0]))

    gt, ga = info['g_TC_rad_per_us'], info['g_AC_rad_per_us']
    interaction = gt*(b.a.dag()*b.sm+b.a*b.sm.dag()) + ga*(b.a*b.Rplus+b.a.dag()*b.Rplus.dag())
    terms = [interaction, [b.Pe, dt], [b.Pr0, d0], [b.Pr1, d1], [b.Pwrong, dw]]
    for op, coeff in ((b.Bprep, prep_drive), (b.Bmap0, map0), (b.Bmap1, map1)):
        terms.extend([[op, coeff], [op.dag(), lambda t, f=coeff: np.conj(f(t))]])
    h = qt.QobjEvo(terms)
    points = [np.linspace(0, mapping[1], 1401), np.array([
        *prep, start, start+ramp, end-ramp, end, *mapping])]
    if include_storage and b.STORAGE:
        points.append(np.linspace(mapping[1], finish, 301))
    info.update(hold_us=hold, ramp_us=ramp, mapping_end_us=mapping[1],
                stages={'Preparation': prep, 'Simultaneous exchange': (start, end),
                        'Optical mapping': mapping, 'Storage': (mapping[1], finish)})
    return h, np.unique(np.concatenate(points)), info, flux, voltage, field


def simulate(b, alpha, beta, noisy=True, include_storage=True, storage_solver=None):
    h, times, info, *_ = model(b, include_storage)
    initial = qt.tensor((alpha*b.g+beta*b.e).unit(), qt.basis(b.NC, 0), b.h0)
    options = dict(store_states=True, atol=1e-11, rtol=1e-9, method='dop853',
                   max_step=min(b.MAX_STEP, b._SETTINGS['SIM_RAMP']/4), nsteps=100000)
    dynamics_times = times[times <= info['mapping_end_us']]
    if noisy:
        result = qt.mesolve(h, qt.ket2dm(initial), dynamics_times, c_ops=b.C_OPS, options=options)
    else:
        result = qt.sesolve(h, initial, dynamics_times, options=options)
    states = list(result.states)
    storage_times = times[times > info['mapping_end_us']]-info['mapping_end_us']
    if len(storage_times):
        parked = h(info['mapping_end_us'])
        if noisy:
            solver = storage_solver or qt.MESolver(parked, b.C_OPS, options={
                'method': 'diag', 'normalize_output': False, 'store_states': True})
            held = solver.run(states[-1], np.r_[0., storage_times]).states
        else:
            held = qt.sesolve(parked, states[-1], np.r_[0., storage_times],
                             options=options).states
        states.extend(held[1:])
    return times, states, info


def run(b, save=False):
    """Panel output: populations, controls, joint fidelity and logical inputs."""
    _, calibration, _ = simulate(b, 1, 1, noisy=False, include_storage=False)
    phase = float(np.angle(calibration[-1].ptrace(2)[0, 1]))
    rotation = b.memory_rotation(phase)
    theta, phi = np.deg2rad([b._SETTINGS['INPUT_THETA'], b._SETTINGS['INPUT_PHI']])
    alpha, beta = np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)
    h, _, setup, *_ = model(b)
    storage_solver = (qt.MESolver(h(setup['mapping_end_us']), b.C_OPS, options={
        'method': 'diag', 'normalize_output': False, 'store_states': True}) if b.STORAGE else None)
    times, states, info = simulate(b, alpha, beta, storage_solver=storage_solver)
    target = qt.tensor(b.g, qt.basis(b.NC, 0), rotation.dag()*(alpha*b.h0+beta*b.h1))
    fidelity = np.real(qt.expect(target.proj(), states))
    operators = (b.Pe, b.n, b.Ph0, b.Ph1, b.Pr0, b.Pr1, b.Plost, b.Pwrong)
    names = ('transmon_excited', 'cavity_photons', 'h0', 'h1', 'r0', 'r1', 'loss', 'wrong')
    pops = dict(zip(names, np.real(qt.expect(operators, states))))
    basis = []
    output_states = [qt.tensor(b.g, qt.basis(b.NC, 0), atom) for atom in (b.h0, b.h1)]
    for label, x, y in (('g to h0', 1, 0), ('e to h1', 0, 1), ('+ to +', 1/np.sqrt(2), 1/np.sqrt(2))):
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
    print('Simultaneous shared-cavity resonance (nominal controls; no tolerance scan).', flush=True)
    print(f"Resonant flux={info['resonant_flux']:.9g}; voltage={info['resonant_voltage_V']:.9g} V", flush=True)
    print(f"g_TC/2pi={info['g_TC_rad_per_us']/b.TWOPI:.6g} MHz; g_AC/2pi={info['g_AC_rad_per_us']/b.TWOPI:.6g} MHz", flush=True)
    print(f"Hold={info['hold_us']:.6g} us; ideal flat-hold transfer limit={info['ideal_transfer_limit']:.6f}", flush=True)
    print(f'Joint fidelity after mapping={fidelity[index]:.6f}; after storage={fidelity[-1]:.6f}', flush=True)
    _, _, _, flux, voltage, field = model(b)
    traces = {
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
    for name, label in zip(names, ('Transmon e', 'Cavity photons', 'h0', 'h1',
                                  'r0', 'r1', 'Loss', 'Wrong Rydberg')):
        population_axis.plot(times[transfer], pops[name][transfer], label=label)
    population_axis.set(ylabel='Population / photon number', title='Write transfer dynamics')
    population_axis.legend(fontsize=8, ncol=2)
    held_coherence = float(2*abs(states[-1].ptrace(2)[0, 1]))
    rows = [
        ['Total write time (preparation through mapping)', f"{info['mapping_end_us']*1000:.2f} ns"],
        ['Storage hold after writing', f'{b.STORAGE:g} µs'],
        ['Total write + storage duration', f"{info['mapping_end_us']+b.STORAGE:.6g} µs"],
        ['Simultaneous exchange pulse (including ramps)', f"{(info['hold_us']+2*info['ramp_us'])*1000:.2f} ns"],
        ['Resonant hold', f"{info['hold_us']*1000:.2f} ns"],
        ['Ideal flat-hold exchange limit', f"{info['ideal_transfer_limit']:.2%}"],
        ['Joint fidelity: g → h0', f"{basis[0]['joint_fidelity_after_storage']:.2%}"],
        ['Joint fidelity: e → h1', f"{basis[1]['joint_fidelity_after_storage']:.2%}"],
        ['Joint fidelity: + → +', f"{basis[2]['joint_fidelity_after_storage']:.2%}"],
        ['Selected input: fidelity before storage', f'{fidelity[index]:.2%}'],
        ['Selected input: fidelity after storage', f'{fidelity[-1]:.2%}'],
        ['Selected input: atomic coherence 2|ρ(h0,h1)|', f'{held_coherence:.6f}'],
        ['Selected input: residual transmon e', f"{pops['transmon_excited'][-1]:.3e}"],
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
    fig.suptitle('Simultaneous cavity resonance: transmon to atom (nominal)')
    fig.canvas.manager.set_window_title('Simultaneous cavity resonance')
    summary.update(displayed_metrics=rows, selected_coherence_after_storage=held_coherence,
                   populations_after_storage={name: float(values[-1]) for name, values in pops.items()})
    if b._ASSISTANT_OUTPUT:
        write_report(b._ASSISTANT_OUTPUT, 'write', b._SETTINGS,
                     dict(protocol='simultaneous_cavity_resonance', noise_enabled=True,
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
