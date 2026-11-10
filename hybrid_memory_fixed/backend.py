"""Effective transmon-cavity-Rydberg memory model with flux and voltage control.

Units: time us, ordinary frequency MHz, angular frequency rad/us.
Illustrative parameters; not a calibrated 87Rb level scheme.
Includes a coherent unwanted Rydberg level and preparation leakage scan.
Run: python backend.py  (opens the parameter panel)
Direct default plots: python backend.py --run
Summary only: python backend.py --paper-figure
Check: python backend.py --check
"""
import argparse
import os
import sys
import json
from functools import lru_cache
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import qutip as qt
from pathlib import Path
from simulation_output import trace, write_report
import tolerance_bands
import simulation_progress as progress

_ASSISTANT_OUTPUT = None

# Static experimental errors. Only the tolerance scan changes these; all
# nominal controls, timings, couplings and optical carrier phases stay fixed.
TOLERANCE_OFFSETS = dict(FC=0.0, FT=0.0, FA=0.0, FLUX=0.0, EFIELD=0.0)

def cavity_error(t):
    return 2 * np.pi * TOLERANCE_OFFSETS["FC"]


_config_parser = argparse.ArgumentParser(add_help=False)
_config_parser.add_argument("--config")
_config_args, _ = _config_parser.parse_known_args()
_SETTINGS = {}
if _config_args.config:
    try:
        with Path(_config_args.config).open(encoding="utf-8") as config_file:
            _SETTINGS = json.load(config_file)
        if not isinstance(_SETTINGS, dict):
            raise ValueError("settings must be a JSON object")
    except (OSError, ValueError) as exc:
        _config_parser.error(f"Cannot load config {_config_args.config!r}: {exc}")

from hybrid_settings import validate, coupling_strengths, derived, calculated_transmon_frequency, calculated_swap_field, transmon_operating_point
_SETTINGS = validate(_SETTINGS)
_COUPLINGS = coupling_strengths(_SETTINGS)

def parameter(name, default):
    return _SETTINGS.get(name, default)

TWOPI = 2 * np.pi

# Device parameters: EJ/h and EC/h, in MHz.
FC = parameter("FC", 5000.0) # actual cavity frequency in the Hamiltonian
F_T_INTERACTION = transmon_operating_point(_SETTINGS)["transmon_interaction_frequency_MHz"]
EC = parameter("EC", 250.0) # transmon charging energy
EJ_MAX = parameter("EJ_MAX", 20000.0) # maximum JJ energy
ASYMMETRY = parameter("ASYMMETRY", 0.10) # asymmetry between JJ
F_T_PARK = calculated_transmon_frequency(_SETTINGS, _SETTINGS["PHI_PARK"])
# Derived g/(2*pi) in MHz -> angular coupling in rad/us for the Hamiltonian.
G_TC = TWOPI * _COUPLINGS["transmon_coupling_MHz"]
G_AC = TWOPI * _COUPLINGS["atom_coupling_MHz"]

# Example low-field Stark coefficients, defined by shift/h = K*E^2.
# K is MHz/(V/cm)^2, already INCLUDING the conventional -alpha/(2h).
# These are placeholders, not parameters of a selected rubidium pair.
F_A_ZERO = parameter("F_A_ZERO", 5150.0)
K_R0 = parameter("K_R0", -2.0)
K_R1 = parameter("K_R1", -12.0)
GAP_CM = parameter("GAP_CM", 0.10)
STRAY_FIELD = parameter("STRAY_FIELD", 0.20)
E_PARK = parameter("E_PARK", 0.0)
# Applied control voltage is independent of the cavity frequency.
E_SWAP = calculated_swap_field(_SETTINGS)
V_SWAP = _SETTINGS["V_SWAP"]

# Coherence times (us), phenomenological Markovian rates.
T1_T = parameter("T1_T", 100.0)
TPHI_T = parameter("TPHI_T", 150.0)
T1_C = parameter("T1_C", 200.0)
TAU_R0 = parameter("TAU_R0", 100.0)
TAU_R1 = parameter("TAU_R1", 100.0)
# Paper example: 43P3/2 is 1320 MHz above 43P1/2 (PDF p. 8).
# Only the neighbor splitting is borrowed; the full atomic pair is illustrative.
F_WRONG_SPLIT = parameter("F_WRONG_SPLIT", 1320.0)
REL_WRONG_DIPOLE = parameter("REL_WRONG_DIPOLE", 1.0)
K_WRONG = parameter('K_WRONG', K_R0)         # placeholder: replace with this level's Stark coefficient
TAU_WRONG = parameter("TAU_WRONG", 100.0)
TPHI_R = parameter("TPHI_R", 100.0)
TPHI_H = parameter("TPHI_H", 100000.0)

# Smooth optical pulses: integral Omega(t) dt = pi.
OMEGA = TWOPI * parameter('OMEGA', 20.0)
T_OPT = TWOPI / OMEGA  # sin^2 envelope averages to 1/2
RAMP = parameter("RAMP", 0.0005)
GAP = parameter("GAP", 0.01)
STORAGE = parameter("STORAGE", 5.0)
MAX_STEP = parameter("MAX_STEP", 0.0005)

PHI_PARK = _SETTINGS["PHI_PARK"]
# Applied interaction flux sets the resonance independently of FC.
PHI_SWAP = _SETTINGS["PHI_SWAP"]

def transmon_frequency(phi):
    ej = EJ_MAX * np.sqrt(
        np.cos(np.pi * phi)**2
        + ASYMMETRY**2 * np.sin(np.pi * phi)**2
    )
    return np.sqrt(8 * EC * ej) - EC

# Calibrated multipliers stretch the rounded TC and AC pulses independently.
# Rounded controls reach the interaction point only at the pulse midpoint;
# pi/(2*g) is a reference time, not a resonant flat hold or guaranteed swap.
PREP = (0.0, T_OPT)
TC_DURATION = _SETTINGS["TC_PULSE_SCALE"] * (2 * RAMP + np.pi / (2 * G_TC))
AC_DURATION = _SETTINGS["AC_PULSE_SCALE"] * (2 * RAMP + np.pi / (2 * G_AC))
TC = (PREP[1] + GAP, PREP[1] + GAP + TC_DURATION)
AC = (TC[1] + GAP, TC[1] + GAP + AC_DURATION)
MAP = (AC[1] + GAP, AC[1] + GAP + T_OPT)
T_END = MAP[1] + STORAGE

def smooth_window(t, start, end):
    """One sin-squared excursion, with zero value and slope at both ends.

    This finite pulse has a nonzero spectral width; it is not a continuous
    single-frequency drive. The full interval sets its rise and fall time.
    """
    if t <= start or t >= end:
        return 0.0
    return float(np.sin(np.pi * (t - start) / (end - start))**2)

def optical_envelope(t, start, end):
    if start < t < end:
        return OMEGA * np.sin(np.pi * (t - start) / (end - start))**2
    return 0.0

def flux(t):
    return PHI_PARK + (PHI_SWAP - PHI_PARK) * smooth_window(t, *TC) + TOLERANCE_OFFSETS["FLUX"]

def electrode_voltage(t):
    # E = V/d + E_stray; park voltage compensates the assumed static stray field.
    desired_field = E_PARK + (E_SWAP - E_PARK) * smooth_window(t, *AC)
    return GAP_CM * (desired_field - STRAY_FIELD)

def electric_field(t):
    return electrode_voltage(t) / GAP_CM + STRAY_FIELD + TOLERANCE_OFFSETS["EFIELD"]

# Hilbert space: two-level transmon, 3 cavity states, 6 atomic states.
# Basis: h0, h1, r0, r1, loss, wrong-Rydberg.
NC = parameter("NC", 3)
NA = 6
g, e = qt.basis(2, 0), qt.basis(2, 1)
h0, h1, r0, r1, lost, wrong = [qt.basis(NA, j) for j in range(NA)]
IT, IC, IA = qt.qeye(2), qt.qeye(NC), qt.qeye(NA)

def A(target, source):
    return qt.tensor(IT, IC, target * source.dag())

sm = qt.tensor(g * e.dag(), IC, IA)
a = qt.tensor(IT, qt.destroy(NC), IA)
Pe = sm.dag() * sm
n = a.dag() * a
Ph0, Ph1, Pr0, Pr1, Plost = [A(s, s) for s in (h0, h1, r0, r1, lost)]
Pwrong = A(wrong, wrong)
Zt = qt.tensor(e * e.dag() - g * g.dag(), IC, IA)
Zr, Zh = Pr1 - Pr0, Ph1 - Ph0

# Rotating frame: cavity frequency removed; hyperfine reference energies
# removed separately; lower Rydberg reference is its zero-field energy.
# Thus upper Rydberg reference offset is F_A_ZERO - FC.
def delta_t(t):
    return TWOPI * (transmon_frequency(flux(t)) + TOLERANCE_OFFSETS["FT"] - FC)

def delta_r0(t):
    return TWOPI * K_R0 * electric_field(t)**2

def delta_r1(t):
    return TWOPI * (F_A_ZERO + TOLERANCE_OFFSETS["FA"] - FC + K_R1 * electric_field(t)**2)

def delta_wrong(t):
    return TWOPI * (F_WRONG_SPLIT + K_WRONG * electric_field(t)**2)

def prep_coeff(t):
    return 0.5 * optical_envelope(t, *PREP) * np.exp(-1j * D_R0_PARK * (t-PREP[0]))

def prep_conj(t):
    return np.conj(prep_coeff(t))

# Mapping occurs at the compensated park field. The optical carrier
# phases below keep each effective optical arm resonant in this frame.
D_R0_PARK = TWOPI * K_R0 * E_PARK**2
D_R1_PARK = TWOPI * (F_A_ZERO - FC + K_R1 * E_PARK**2)

def map0_coeff(t):
    return 0.5 * optical_envelope(t, *MAP) * np.exp(
        1j * D_R0_PARK * (t - MAP[0])
    )

def map1_coeff(t):
    return 0.5 * optical_envelope(t, *MAP) * np.exp(
        1j * D_R1_PARK * (t - MAP[0])
    )

def map0_conj(t):
    return np.conj(map0_coeff(t))

def map1_conj(t):
    return np.conj(map1_coeff(t))

# Crucial change: both physical couplings remain present at ALL times.
Rplus = A(r1, r0)
H_coupling = G_TC * (a.dag() * sm + a * sm.dag()) + G_AC * (
    a * Rplus + a.dag() * Rplus.dag()
)
# The same laser drives the desired and undesired optical transitions.
# Hermitian-conjugate terms also allow stimulated return from the wrong state.
# The reverse r0->h0 mapping pulse can therefore excite h0->wrong as well.
# Cavity couplings from the wrong state are omitted until specified.
Bprep = A(r0, h0) + REL_WRONG_DIPOLE * A(wrong, h0)
Bmap0 = A(h0, r0) + REL_WRONG_DIPOLE * A(h0, wrong)
Bmap1 = A(h1, r1)
H = qt.QobjEvo([ #construction of the hamiltonian 
    H_coupling, # contrains the transmon-cavity and atom-cavity interactions
    [n, cavity_error], # cavity drift in the fixed nominal rotating frame
    [Pe, delta_t], # time dependent transmon excitation energy offset
    [Pr0, delta_r0], # time dependent Rydberg energy offset
    [Pr1, delta_r1],# time dependent Rydberg energy offset
    [Pwrong, delta_wrong], # time dependent unwanted state
    [Bprep, prep_coeff], # Preparation laser interaction
    [Bprep.dag(), prep_conj], 
    [Bmap0, map0_coeff], # Mapping laser interaction involving h0
    [Bmap0.dag(), map0_conj],
    [Bmap1, map1_coeff], # Mapping laser interaction involtving
    [Bmap1.dag(), map1_conj],
])

C_OPS = [
    np.sqrt(1 / T1_T) * sm,
    np.sqrt(1 / T1_C) * a,
    np.sqrt(1 / TAU_R0) * A(lost, r0),
    np.sqrt(1 / TAU_R1) * A(lost, r1),
    np.sqrt(1 / TAU_WRONG) * A(lost, wrong),
    np.sqrt(1 / (2 * TPHI_T)) * Zt,
    np.sqrt(1 / (2 * TPHI_R)) * Zr,
    np.sqrt(1 / (2 * TPHI_H)) * Zh,
]

def time_grid(include_storage=True):
    # Include every stage endpoint explicitly in the recorded output.
    points = [np.linspace(0, MAP[1], 1801)]
    for interval in (PREP, TC, AC, MAP):
        points.append(np.asarray(interval))
    points.append(np.asarray([sum(TC) / 2, sum(AC) / 2]))
    if include_storage:
        points.append(np.linspace(MAP[1], T_END, 301))
    return np.unique(np.concatenate(points))

def simulate(alpha, beta, noisy=True, include_storage=True):
    q = (alpha * g + beta * e).unit()
    psi = qt.tensor(q, qt.basis(NC, 0), h0)
    ts = time_grid(include_storage)
    options = {
        "store_states": True, "atol": 1e-11, "rtol": 1e-9, "method": "dop853",
        "max_step": MAX_STEP, "nsteps": 100000,
    }
    if noisy:
        result = qt.mesolve(H, qt.ket2dm(psi), ts, c_ops=C_OPS, options=options)
    else:
        result = qt.sesolve(H, psi, ts, options=options)
    return ts, result.states

def calibrated_phase():
    # A single nominal, noise-free |+> calibration fixes the known
    # dynamical phase. This same correction is used for ALL inputs.
    # It does not correct lost population or decoherence.
    _, states = simulate(1, 1, noisy=False, include_storage=False)
    rho_a = states[-1].ptrace(2)
    return float(np.angle(rho_a[0, 1]))

def memory_rotation(phase):
    # Virtual Z / calibrated phase reference on the memory qubit.
    return IA + (np.exp(1j * phase) - 1) * h1 * h1.dag()

def overlap(rho_a, target, U):
    return float(np.real(qt.expect(
        qt.ket2dm(target), U * rho_a * U.dag()
    )))

def results_table(ax, headers, rows, widths=None, bbox=(0, 0.20, 1, 0.58), fontsize=11):
    """Render a shared numeric summary without plot axes."""
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                     cellLoc="center", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dc")
        if row == 0:
            cell.set_facecolor("#edf1f5")
            cell.set_text_props(weight="bold")
        elif row % 2:
            cell.set_facecolor("#fafbfc")
    return table


def basis_probabilities(mode, roundtrip=False, source="hyperfine", phase=None, storage=False):
    """Exclusive joint outcomes and coherent fidelity for g/e/+ inputs.

    Columns are probabilities of orthogonal joint basis outputs. The separate
    target-state fidelity also tests coherence for the equal superposition.
    """
    outputs = ([qt.tensor(g, qt.basis(NC, 0), h) for h in (h0, h1)]
               if mode == "write" else
               [qt.tensor(t, qt.basis(NC, 0), h0) for t in (g, e)])
    labels, rows, fidelities = [], [], []
    if phase is None:
        phase = calibrated_phase() if mode == "write" else read_phase(source, roundtrip)
    u = (qt.tensor(IT, IC, memory_rotation(phase)) if mode == "write" else
         qt.tensor(IT + (np.exp(1j * phase) - 1) * e * e.dag(), IC, IA))
    cases = (True,) if roundtrip else (False,)
    for stored in cases:
        for index, (alpha, beta) in enumerate(((1, 0), (0, 1), (1/np.sqrt(2), 1/np.sqrt(2)))):
            print(f"Calculating {mode} basis input {index}"
                  + (" after write/store" if stored else "") + "…", flush=True)
            if mode == "write":
                _, states = simulate(alpha, beta, include_storage=False)
                label = ("g → h0", "e → h1", "+ → +")[index]
            else:
                if stored:
                    _, write_states = simulate(alpha, beta, include_storage=True)
                    initial = write_states[-1]
                else:
                    initial = stored_retrieval_initial(alpha, beta, source)
                _, states, _, _, _ = evolve_read(initial, source=source)
                label = (("h0 → g", "h1 → e", "+ → +") if source == "hyperfine" else
                         ("r0 → g", "r1 → e", "+ → +"))[index]
                if roundtrip:
                    label = ("g → g", "e → e", "+ → +")[index] + "\nAfter write/store"
            final = states[-1]
            if mode == "write" and storage:
                final = hold_memory(final)
            probs = [float(np.real(qt.expect(qt.ket2dm(k), final))) for k in outputs]
            wrong = float(np.real(qt.expect(Pwrong, final)))
            row = [*probs, wrong, 1-sum(probs)-wrong]
            assert min(row) >= -1e-7 and abs(sum(row)-1) < 1e-7
            logical = ((alpha * h0 + beta * h1).unit() if mode == "write" else
                       (alpha * g + beta * e).unit())
            target = (qt.tensor(g, qt.basis(NC, 0), logical) if mode == "write" else
                      qt.tensor(logical, qt.basis(NC, 0), h0))
            rho = qt.ket2dm(final) if final.isket else final
            fidelity = float(np.real(qt.expect(qt.ket2dm(target), u * rho * u.dag())))
            assert -1e-7 <= fidelity <= 1+1e-7
            labels.append(label)
            rows.append(row)
            fidelities.append(fidelity)
            progress.update((index+1)/3, f"{mode.capitalize()} logical input {index+1}/3 complete")
    return labels, np.asarray(rows), np.asarray(fidelities)


def plot_basis_columns(ax, labels, probabilities, title, mode="write", input_label=None):
    """Original 3D input/output columns, with fixed joint-output rows."""
    for i, row in enumerate(probabilities):
        outputs = [row[0], row[1], row[2] + row[3]]
        for j, value in enumerate(outputs):
            ax.bar3d(i-.23, j-.23, 0, .46, .46, max(value, 0),
                     color="#c3ce48" if (i < 2 and j == i) or (i == 2 and j < 2) else "#c89b32",
                     edgecolor="#4b4934", linewidth=.5, shade=True, alpha=.93)
    input_labels = [label.split(" →")[0] for label in labels]
    ax.set_xticks(np.arange(len(labels)), input_labels)
    ax.set_yticks([0, 1, 2], ["g,0,h0", "g,0,h1" if mode == "write" else "e,0,h0", "Other"])
    ax.set(xlabel=input_label or ("Transmon input" if mode == "write" else "Memory input"),
           ylabel="Joint output", zlabel="Probability", zlim=(0, 1), title=title)
    ax.tick_params(labelsize=7, pad=1)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 6
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.zaxis.label.set_size(9)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_box_aspect((4, 4, 3), zoom=1.12)
    ax.view_init(elev=25, azim=-53)


def plot_wrong_population(ax, cases, xlabel):
    for label, times, states in cases:
        values = np.real(qt.expect(Pwrong, states))
        # A display floor keeps zeros visible on a logarithmic axis.
        ax.semilogy(times, np.maximum(values, 1e-16), label=label)
        print(f"{label}: wrong Rydberg final={values[-1]:.6e}, "
              f"sampled peak={values.max():.6e}", flush=True)
    ax.set(xlabel=xlabel, ylabel="Wrong-Rydberg probability",
           title="Wrong Rydberg (display floor 10⁻¹⁶)")
    ax.legend(fontsize=8)


def main(check=False, save=False):
    work=progress.transfer_stages(_SETTINGS)
    progress.update(0, "Calibrating write phase")
    phase = calibrated_phase()
    progress.update(work["calibration"], "Calculating write dynamics")
    U = memory_rotation(phase)

    # Nonzero input phase checks coherent transfer.
    theta = np.deg2rad(parameter("INPUT_THETA", 90.0))
    phi = np.deg2rad(parameter("INPUT_PHI", 60.0))
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    target = (alpha * h0 + beta * h1).unit()
    ts, states = simulate(alpha, beta, noisy=True, include_storage=False)
    atomic_states = [rho.ptrace(2) for rho in states]
    transfer_target = (alpha * g + beta * e).unit()
    fidelity = np.real(qt.expect(atomic_storage_effect(transfer_target, phase), states))
    coherence = np.array([2 * abs(rho[0, 1]) for rho in atomic_states])
    pops = np.array([
        np.real(qt.expect(op, states))
        for op in (Pe, n, Ph0, Ph1, Pr0, Pr1, Plost, Pwrong)
    ])

    mapping_index = np.argmin(abs(ts - MAP[1]))
    print(f"Flux park/swap (Phi/Phi0): {PHI_PARK:.6f}, {PHI_SWAP:.6f}")
    print(f"Voltage park/swap (V): {electrode_voltage(0):.6f}, "
          f"{GAP_CM * (E_SWAP - STRAY_FIELD):.6f}")
    prep_index = np.argmin(abs(ts - PREP[1]))
    print(f"Wrong-state population after preparation: {pops[7, prep_index]:.6e}")
    print(f"Peak wrong-state population (sampled): {pops[7].max():.6e}")
    print(f"Final wrong-state population: {pops[7, -1]:.6e}")
    print(f"Mapping joint transfer fidelity: {fidelity[mapping_index]:.6f}")
    print(f"Final joint transfer fidelity: {fidelity[-1]:.6f}")
    print(f"Final 2|rho_h0,h1|: {coherence[-1]:.6f}")
    print(f"Calibrated memory phase (rad): {phase:.6f}")

    # Structural checks: trace, Hermiticity, and population bounds.
    assert max(abs(rho.tr() - 1) for rho in states) < 1e-6
    assert (states[-1] - states[-1].dag()).norm() < 1e-6
    print(f"Minimum final eigenvalue: {states[-1].eigenenergies().min():.3e}")
    assert states[-1].eigenenergies().min() > -1e-6
    assert np.all((fidelity > -1e-6) & (fidelity < 1 + 1e-6))
    if check:
        # Reuse the calibrated phase for a distinct |+i> input.
        _, test_states = simulate(1, 1j, noisy=False, include_storage=False)
        test_target = (h0 + 1j * h1).unit()
        test_f = overlap(test_states[-1].ptrace(2), test_target, U)
        print(f"Ideal |+i> overlap with same phase correction: {test_f:.6f}")
        print("Physical-state checks passed.")
        return

    progress.update(work["selected"], "Calculating storage")
    held_state = hold_memory(states[-1])
    held_fidelity = float(np.real(qt.expect(atomic_storage_effect(transfer_target, phase), held_state)))
    with progress.scope(work["storage"], work["basis"], "Write logical inputs"):
        labels, probabilities, input_fidelities = basis_probabilities("write", phase=phase, storage=True)
    with progress.scope(work["basis"], .95, "Write tolerance scenarios"):
        bands = tolerance_bands.compute(sys.modules[__name__], _SETTINGS, "write", alpha, beta,
                                        phase, ts, pops, fidelity, held_fidelity,
                                        nominal_state=held_state, nominal_basis=input_fidelities)
    progress.update(.95, "Building write figure")
    summary_row = 2
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(summary_row + 1, 2)
    control_axis = fig.add_subplot(grid[0, 0])
    voltage_axis = control_axis.twinx()
    detuning_axis = fig.add_subplot(grid[0, 1])
    population_axis = fig.add_subplot(grid[1, :])
    performance_axis = fig.add_subplot(grid[summary_row, 0])
    basis_axis = fig.add_subplot(grid[summary_row, 1], projection="3d")
    x = np.linspace(0, MAP[1], 2501)
    flux_line, = control_axis.plot(x, [flux(t) for t in x],
                                   color="tab:blue", label="Flux (left axis)")
    voltage_line, = voltage_axis.plot(x, [electrode_voltage(t) for t in x],
                                     color="tab:orange", linestyle="--",
                                     label="Voltage (right axis)")
    control_axis.set_ylabel("Flux / flux quantum", color="tab:blue")
    voltage_axis.set_ylabel("Electrode voltage (V)", color="tab:orange")
    control_axis.tick_params(axis="y", colors="tab:blue")
    voltage_axis.tick_params(axis="y", colors="tab:orange")
    control_axis.spines["left"].set_color("tab:blue")
    voltage_axis.spines["right"].set_color("tab:orange")
    control_axis.set_title("Flux and voltage control")
    voltage_axis.legend(handles=[flux_line, voltage_line], loc="upper center",
                        bbox_to_anchor=(0.5, 0.88), ncol=2, fontsize=8,
                        framealpha=1)
    detuning_axis.plot(x, [delta_t(t) / TWOPI for t in x],
                       color="tab:blue", label="Transmon")
    detuning_axis.plot(
        x, [(delta_r1(t) - delta_r0(t)) / TWOPI for t in x],
        color="tab:orange", linestyle="--", label="Atom"
    )
    detuning_axis.axhline(0, color="gray", linewidth=.7, alpha=.5)
    detuning_axis.set(title="Detuning from cavity", ylabel="Detuning (MHz)")
    detuning_axis.legend(fontsize=8)
    transfer = ts <= MAP[1]
    for line, label in zip(pops, ("Transmon e", "Cavity photons", "h0", "h1",
                                  "r0", "r1", "Loss", "Wrong Rydberg")):
        population_axis.plot(ts[transfer], line[transfer], label=label)
    population_axis.set(ylabel="Population / photon number", title="Write transfer dynamics")
    population_axis.legend(fontsize=8, ncol=2)
    held_coherence = 2 * abs(held_state.ptrace(2)[0, 1])
    held_pops = np.real([qt.expect(op, held_state)
                         for op in (Pe, n, Ph0, Ph1, Pr0, Pr1, Plost, Pwrong)])
    print(f"Joint fidelity AFTER {STORAGE:g} us storage: {held_fidelity:.6f}")
    performance_axis.set_axis_off()
    performance_axis.set_title("Write + storage summary")
    summary_rows = [
        ["Total write time (preparation through mapping)", f"{MAP[1] * 1000:.2f} ns"],
        ["Storage hold after writing", f"{STORAGE:g} µs"],
        ["Total write + storage duration", f"{MAP[1] + STORAGE:.6g} µs"],
        ["Transmon–cavity rounded pulse", f"{(TC[1] - TC[0]) * 1000:.2f} ns"],
        ["Cavity–atom rounded pulse", f"{(AC[1] - AC[0]) * 1000:.2f} ns"],
        ["Joint fidelity: g → h0", f"{input_fidelities[0]:.2%}"],
        ["Joint fidelity: e → h1", f"{input_fidelities[1]:.2%}"],
        ["Joint fidelity: + → +", f"{input_fidelities[2]:.2%}"],
        ["Selected input: fidelity before storage", f"{fidelity[-1]:.2%}"],
        ["Selected input: fidelity after storage", f"{held_fidelity:.2%}"],
        ["Selected input: atomic coherence 2|ρ(h0,h1)|", f"{held_coherence:.6f}"],
        ["Selected input: residual transmon e", f"{held_pops[0]:.3e}"],
        ["Selected input: residual mean cavity photons", f"{held_pops[1]:.3e}"],
        ["Selected input: residual r0 + r1 population", f"{held_pops[4] + held_pops[5]:.3e}"],
        ["Selected input: atomic loss population", f"{held_pops[6]:.3e}"],
        ["Selected input: wrong-Rydberg population", f"{held_pops[7]:.3e}"],
    ]
    summary_rows = tolerance_bands.table_rows(summary_rows, bands, "write")
    if bands:
        tolerance_bands.draw(bands, control_axis, detuning_axis, population_axis)
    table = results_table(performance_axis, ["Metric", "Nominal", "With tolerances"],
                          summary_rows, widths=[.58, .16, .26],
                          bbox=(0, 0.04, 1, 0.94), fontsize=7.5)
    for row in range(1, len(summary_rows) + 1):
        table[row, 0].set_text_props(ha="left")
    if save:
        # Retain the optional time-resolved CSV even though the figure now
        # displays final performance rather than final-target overlap traces.
        data = transfer_fidelity_data("write")
        csv_output = Path(__file__).with_name("hybrid_transfer_write_fidelity.csv")
        np.savetxt(csv_output, data, delimiter=",", comments="",
                   header="transfer_time_us,ground_joint_fidelity,excited_joint_fidelity,plus_joint_fidelity")
        print(f"Saved {csv_output}", flush=True)
    plot_basis_columns(basis_axis, labels, probabilities, "Write outputs after storage" + (" (nominal)" if bands else ""))
    basis_axis.set_box_aspect((4, 4, 3), zoom=.98)
    basis_axis.xaxis.labelpad = -5
    fig.supxlabel(f"Selected input: θ={np.rad2deg(theta):g}°, φ={np.rad2deg(phi):g}°.\n"
                  f"Initial transmon populations: g = {100 * np.cos(theta / 2)**2:.2f}%, "
                  f"e = {100 * np.sin(theta / 2)**2:.2f}%.",
                  fontsize=9)
    for ax in (control_axis, detuning_axis, population_axis):
        ax.set_xlabel("Time (us)")
        for start, end in (PREP, TC, AC, MAP):
            ax.axvline(start, color="gray", alpha=0.15)
            ax.axvline(end, color="gray", alpha=0.15)
    fig.suptitle("Transmon to atom" + (" — shaded sampled tolerance ranges" if bands else ""))
    fig.canvas.manager.set_window_title("Transmon to atom")
    if _ASSISTANT_OUTPUT:
        names = ("transmon_e", "cavity_mean_photons", "atom_h0", "atom_h1",
                 "atom_r0", "atom_r1", "atom_loss", "atom_wrong_rydberg")
        write_report(_ASSISTANT_OUTPUT, "write", _SETTINGS,
            metadata={"noise_enabled": True, "input_theta_degrees": float(np.rad2deg(theta)),
                      "input_phi_degrees": float(np.rad2deg(phi)), "phase_correction_rad": float(phase),
                      "stages_us": dict(preparation=PREP, transmon_cavity=TC, atom_cavity=AC, mapping=MAP),
                      "storage_duration_us": STORAGE,
                      "time_origin": "Beginning of writing; dynamics end before the storage hold.",
                      "units": "time us; detuning MHz; flux Phi/Phi0; voltage V; populations/fidelity/coherence dimensionless; cavity is mean photon number",
                      "fidelity_definition": "Unconditional joint target overlap; fixed noiseless phase; no postselection.",
                      "basis_evaluation": "After writing AND storage; inputs are original transmon g/e/+"},
            summary={"tolerance_ranges": bands["summary"] if bands else None,
                     "displayed_metrics": summary_rows, "selected_fidelity_before_storage": float(fidelity[-1]),
                     "selected_fidelity_after_storage": held_fidelity,
                     "selected_coherence_after_storage": float(held_coherence),
                     "populations_after_storage": dict(zip(names, held_pops))},
            traces={**tolerance_bands.report_traces(bands),
                    "selected_input_dynamics": trace(ts, joint_fidelity=fidelity,
                        atomic_coherence_2abs_rho_h0_h1=coherence, **dict(zip(names, pops))),
                    "controls": trace(x, flux_Phi_over_Phi0=[flux(t) for t in x],
                        electrode_voltage_V=[electrode_voltage(t) for t in x],
                        transmon_detuning_MHz=[delta_t(t)/TWOPI for t in x],
                        atom_detuning_MHz=[(delta_r1(t)-delta_r0(t))/TWOPI for t in x])},
            basis={"input_labels": labels, "joint_output_labels": ["g,0,h0", "g,0,h1", "wrong_rydberg", "remaining_other"],
                   "probabilities": probabilities, "joint_fidelities": input_fidelities})
    output = Path(__file__).with_name("hybrid_controls.png")
    if save:
        fig.savefig(output, dpi=170)
        print(f"Plot saved to {output}")


def isolated_preparation(duration, dipole_ratio=REL_WRONG_DIPOLE,
                         split_mhz=F_WRONG_SPLIT, resolution=1):
    """Closed three-level optical test: |h0>, |r0>, |wrong>.
    Zero field, no cavity or decay, to isolate off-resonant excitation.
    Equal-area sin^2 pi pulses, identical shape to full preparation.
    """
    b, r, w = [qt.basis(3, j) for j in range(3)]
    B = r * b.dag() + dipole_ratio * w * b.dag()
    def drive(t):
        return np.pi / duration * np.sin(np.pi * t / duration)**2
    h = [TWOPI * split_mhz * w * w.dag(), [B + B.dag(), drive]]
    # Resolve the optical detuning and the pulse envelope independently.
    step = min(duration / 100, 1 / (30 * max(abs(split_mhz), 1))) / resolution
    result = qt.sesolve(
        h, b, np.linspace(0, duration, 301),
        e_ops=[b * b.dag(), r * r.dag(), w * w.dag()],
        options={"method": "dop853", "atol": 1e-12, "rtol": 1e-10,
                 "max_step": step, "nsteps": 100000},
    )
    return np.array(result.expect)


def plot_preparation_probabilities(ax):
    """Final populations versus duration of an isolated, equal-area π pulse.

    Vary the peak drive with duration to keep the pulse area fixed, matching
    isolated_preparation; this is a pulse scan, not a time-evolution trace.
    """
    durations = np.linspace(0.00002, 0.003, 81)  # 0.02–3 ns; no zero-duration pulse
    populations = np.asarray([isolated_preparation(t)[:, -1] for t in durations])
    assert np.all(populations >= -1e-8)
    assert np.allclose(populations.sum(axis=1), 1, atol=1e-8)
    ax.plot(durations * 1000, populations[:, 1], color="blue",
            label="Desired Rydberg (r0)")
    ax.plot(durations * 1000, populations[:, 2], color="blue", linestyle="--",
            label="Wrong Rydberg")
    marked = isolated_preparation(0.001)[:, -1]
    ax.plot([1, 1], marked[[1, 2]], linestyle="none", marker="o",
            markerfacecolor="none", markeredgecolor="red", markersize=5)
    ax.set(xlabel="Preparation pulse duration (ns)", ylabel="Final probability",
           xlim=(0, 3), ylim=(-.02, 1.02), title="Rydberg preparation: pulse-duration scan")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="center right", fontsize=8)
    ax.text(.97, .24, f"Splitting: {F_WRONG_SPLIT / 1000:g} GHz\n"
            "Isolated equal-area π pulses", transform=ax.transAxes,
            ha="right", fontsize=8)


def leakage_scan(save=False):
    durations = np.geomspace(0.0002, T_OPT, 35)  # us: 0.2 to 100 ns
    rows = []
    for duration in durations:
        populations = isolated_preparation(duration)
        rows.append([duration * 1000, *populations[:, -1],
                     populations[2].max()])
    rows = np.asarray(rows)

    # Tests isolate this new mechanism from the rest of the protocol.
    no_wrong = isolated_preparation(0.001, dipole_ratio=0)
    assert abs(no_wrong[1, -1] - 1) < 1e-7
    assert no_wrong[2].max() < 1e-12
    short = isolated_preparation(0.001)
    refined = isolated_preparation(0.001, resolution=2)
    assert short[2].min() > -1e-9  # Leakage may vanish for a valid parameter choice.
    assert abs(short[2, -1] - refined[2, -1]) < 1e-7
    assert np.max(abs(short.sum(axis=0) - 1)) < 1e-8

    print("Leakage checks passed: zero-coupling limit, nonzero short-pulse "
          "excitation, normalization, and step refinement.")
    print(f"1 ns pulse: target={short[1,-1]:.6f}, "
          f"wrong at end={short[2,-1]:.6e}, peak wrong={short[2].max():.6f}")
    print(f"{T_OPT*1000:.1f} ns pulse: target={rows[-1,2]:.8f}, "
          f"wrong at end={rows[-1,3]:.6e}")

    output = Path(__file__).with_name("rydberg_leakage_scan")
    if save:
        np.savetxt(output.with_suffix(".csv"), rows, delimiter=",",
                   header="duration_ns,ground_final,target_final,wrong_final,wrong_peak",
                   comments="")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogx(rows[:,0], rows[:,2], label="Desired Rydberg (final)")
    axes[0].semilogx(rows[:,0], rows[:,1], label="Ground state (final)")
    axes[0].semilogx(rows[:,0], rows[:,3], label="Wrong Rydberg (final)")
    axes[0].set(xlabel="Pulse duration (ns)", ylabel="Population",
                title="Equal-area smooth preparation pulses", ylim=(0, 1.03))
    axes[0].legend(fontsize=8)
    axes[1].loglog(rows[:,0], np.maximum(rows[:,3], 1e-18),
                   label="Wrong Rydberg at pulse end")
    axes[1].loglog(rows[:,0], np.maximum(rows[:,4], 1e-18),
                   label="Peak wrong Rydberg")
    axes[1].set(xlabel="Pulse duration (ns)", ylabel="Population",
                title="Off-resonant excitation (log scale)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Illustrative 1320 MHz neighbor; relative dipole amplitude = 1")
    fig.tight_layout()
    if save:
        fig.savefig(output.with_suffix(".png"), dpi=170)
        print(f"Leakage scan saved to {output.with_suffix('.png')}")



def retrieval_model(source="hyperfine"):
    """Build the readout controls. Source can be hyperfine or rydberg.
    Couplings and loss channels remain the same as during writing.
    """
    if source not in ("hyperfine", "rydberg"):
        raise ValueError("source must be hyperfine or rydberg")
    unmap = (0.0, T_OPT) if source == "hyperfine" else None
    start = T_OPT + GAP if unmap else 0.0
    ac = (start, start + (AC[1] - AC[0]))
    tc = (ac[1] + GAP, ac[1] + GAP + (TC[1] - TC[0]))
    cleanup = (tc[1] + GAP, tc[1] + GAP + T_OPT)

    def flux_read(t):
        return PHI_PARK + (PHI_SWAP - PHI_PARK) * smooth_window(t, *tc) + TOLERANCE_OFFSETS["FLUX"]

    def voltage_read(t):
        field = E_PARK + (E_SWAP - E_PARK) * smooth_window(t, *ac)
        return GAP_CM * (field - STRAY_FIELD)

    def field_read(t):
        return voltage_read(t) / GAP_CM + STRAY_FIELD + TOLERANCE_OFFSETS["EFIELD"]

    def dt_read(t):
        return TWOPI * (transmon_frequency(flux_read(t)) + TOLERANCE_OFFSETS["FT"] - FC)

    def d0_read(t):
        return TWOPI * K_R0 * field_read(t)**2

    def d1_read(t):
        return TWOPI * (F_A_ZERO + TOLERANCE_OFFSETS["FA"] - FC + K_R1 * field_read(t)**2)

    def dw_read(t):
        return TWOPI * (F_WRONG_SPLIT + K_WRONG * field_read(t)**2)

    # Bmap operators lower Rydberg states; their Hermitian conjugates
    # supply the memory->Rydberg direction under the same resonant drive.
    def unmap0(t):
        if unmap is None:
            return 0.0
        return 0.5 * optical_envelope(t, *unmap) * np.exp(
            1j * D_R0_PARK * (t - unmap[0])
        )

    def unmap1(t):
        if unmap is None:
            return 0.0
        return 0.5 * optical_envelope(t, *unmap) * np.exp(
            1j * D_R1_PARK * (t - unmap[0])
        )

    def unmap0_conj(t):
        return np.conj(unmap0(t))

    def unmap1_conj(t):
        return np.conj(unmap1(t))

    # After atom->cavity->transmon, the atom should be in r0.
    # Return it to h0 so it need not remain in a short-lived Rydberg state.
    def cleanup_coeff(t):
        return 0.5 * optical_envelope(t, *cleanup) * np.exp(
            -1j * D_R0_PARK * (t-cleanup[0]))

    def cleanup_conj(t):
        return np.conj(cleanup_coeff(t))

    h_read = qt.QobjEvo([
        H_coupling,
        [n, cavity_error],
        [Pe, dt_read], [Pr0, d0_read], [Pr1, d1_read], [Pwrong, dw_read],
        [Bmap0, unmap0], [Bmap0.dag(), unmap0_conj],
        [Bmap1, unmap1], [Bmap1.dag(), unmap1_conj],
        [Bprep, cleanup_coeff], [Bprep.dag(), cleanup_conj],
    ])
    stages = []
    if unmap:
        stages.append(("Memory to Rydberg", unmap))
    stages.extend([("Atom to cavity", ac), ("Cavity to transmon", tc),
                   ("Return atom to h0", cleanup)])
    points = [np.linspace(0, cleanup[1], 1801)]
    points.extend(np.asarray(interval) for _, interval in stages)
    points.append(np.asarray([sum(tc) / 2, sum(ac) / 2]))
    return h_read, np.unique(np.concatenate(points)), stages, flux_read, voltage_read


def evolve_read(initial, noisy=True, source="hyperfine", step=MAX_STEP):
    h_read, ts, stages, flux_read, voltage_read = retrieval_model(source)
    options = {"store_states": True, "atol": 1e-11, "rtol": 1e-9,
               "method": "dop853", "max_step": step, "nsteps": 100000}
    if not noisy and initial.isket:
        result = qt.sesolve(h_read, initial, ts, options=options)
    else:
        rho = qt.ket2dm(initial) if initial.isket else initial
        result = qt.mesolve(h_read, rho, ts, c_ops=C_OPS if noisy else [],
                            options=options)
    return ts, result.states, stages, flux_read, voltage_read


def retrieval_initial(alpha, beta, source="hyperfine"):
    pair = (h0, h1) if source == "hyperfine" else (r0, r1)
    atom = (alpha * pair[0] + beta * pair[1]).unit()
    return qt.tensor(g, qt.basis(NC, 0), atom)


@lru_cache(maxsize=1)
def parked_storage_solver():
    """Reuse the constant parked Lindblad generator for independent inputs."""
    return qt.MESolver(H(MAP[1]), C_OPS, options={
        "method": "diag", "normalize_output": False,
    })


def hold_memory(initial, noisy=True):
    """Evolve the complete joint state at park for STORAGE, without resetting it."""
    if STORAGE == 0:
        return initial
    if not noisy:
        unitary = (-1j * H(MAP[1]) * STORAGE).expm()
        return unitary * initial if initial.isket else unitary * initial * unitary.dag()
    rho = initial.proj() if initial.isket else initial
    return parked_storage_solver().run(rho, [0, STORAGE]).states[-1]


def stored_retrieval_initial(alpha, beta, source="hyperfine", noisy=True):
    """Prepare an atomic input and retain its full state after the storage hold."""
    return hold_memory(retrieval_initial(alpha, beta, source), noisy=noisy)


def read_phase(source="hyperfine", roundtrip=False):
    # Fixed calibration from a nominal |+> input; never fit to noisy data
    # or separately to each unknown input state.
    if roundtrip:
        _, write_states = simulate(1, 1, noisy=False, include_storage=True)
        initial = write_states[-1]
    else:
        initial = stored_retrieval_initial(1, 1, source, noisy=False)
    _, states, _, _, _ = evolve_read(initial, noisy=False, source=source)
    return float(np.angle(states[-1].ptrace(0)[0, 1]))


def read_metrics(states, target, phase):
    # A virtual Z / phase-reference correction, not an extra recovery of
    # lost amplitude. Report both raw and phase-corrected overlaps.
    u = IT + (np.exp(1j * phase) - 1) * e * e.dag()
    raw, corrected, joint, coherence = [], [], [], []
    u_full = qt.tensor(u, IC, IA)
    joint_target = qt.tensor(target, qt.basis(NC, 0), h0)
    for state in states:
        rho = qt.ket2dm(state) if state.isket else state
        rho_t = rho.ptrace(0)
        raw.append(float(np.real(qt.expect(qt.ket2dm(target), rho_t))))
        corrected.append(float(np.real(qt.expect(
            qt.ket2dm(target), u * rho_t * u.dag()
        ))))
        joint.append(float(np.real(qt.expect(
            qt.ket2dm(joint_target), u_full * rho * u_full.dag()
        ))))
        coherence.append(2 * abs(rho_t[0, 1]))
    return tuple(np.asarray(v) for v in (raw, corrected, joint, coherence))


def retrieval_effect(target, phase, noisy=True):
    """Pull a corrected joint target back through the fixed read channel.

    Tr[E_read(rho) P] = Tr[rho E_read^dagger(P)]. This avoids repeating
    readout for every storage duration. Effects must not be trace-normalized.
    """
    h_read, times, _, _, _ = retrieval_model("hyperfine")
    rotation = IT + (np.exp(1j * phase) - 1) * e * e.dag()
    u = qt.tensor(rotation, IC, IA)
    projector = qt.ket2dm(qt.tensor(target, qt.basis(NC, 0), h0))
    effect = u.dag() * projector * u
    adjoint = qt.liouvillian(h_read, C_OPS if noisy else []).dag()
    duration = times[-1]

    def reverse_generator(t):
        return adjoint(duration - t)

    result = qt.mesolve(qt.QobjEvo(reverse_generator), effect,
                        [0.0, duration], options={
                            "method": "dop853", "max_step": MAX_STEP,
                            "atol": 1e-11, "rtol": 1e-9,
                            "nsteps": 100000, "normalize_output": False,
                        })
    return result.states[-1]


def direct_transmon_fidelity(elapsed):
    """Six-state mean for isolated transmon T1/pure-dephasing storage.

    Deterministic parked-frequency precession is removed by its rotating frame.
    """
    elapsed = np.asarray(elapsed)
    t2_inverse = 1 / (2 * T1_T) + 1 / TPHI_T
    return (3 + 2 * np.exp(-elapsed * t2_inverse)
            + np.exp(-elapsed / T1_T)) / 6


def atomic_storage_effect(target, phase):
    """Joint write/store target: transmon g, vacuum, input encoded in h0/h1.

    Pull back the fixed memory phase correction, without conditioning on
    successful transfer. Residual transmon/cavity excitation counts as error.
    """
    atom = (target[0, 0] * h0 + target[1, 0] * h1).unit()
    joint = qt.tensor(g, qt.basis(NC, 0), atom)
    rotation = qt.tensor(IT, IC, memory_rotation(phase))
    return rotation.dag() * joint.proj() * rotation


@lru_cache(maxsize=4)
def transfer_fidelity_data(mode, source="hyperfine"):
    """Time, then unconditional joint fidelities for logical 0, 1 and +.

    Each direction starts with an independently prepared source qubit and
    an empty destination. Read inputs undergo STORAGE before readout.
    One noiseless phase calibration serves all inputs.
    The + projector includes coherence, unlike a population-only score.
    """
    if mode not in ("write", "read"):
        raise ValueError("mode must be write or read")
    phase = calibrated_phase() if mode == "write" else read_phase(source)
    curves = []
    for target in (g, e, (g + e).unit()):
        alpha, beta = target[0, 0], target[1, 0]
        if mode == "write":
            times, states = simulate(alpha, beta, include_storage=False)
            values = np.real(qt.expect(atomic_storage_effect(target, phase), states))
        else:
            initial = stored_retrieval_initial(alpha, beta, source)
            times, states, _, _, _ = evolve_read(initial, source=source)
            values = read_metrics(states, target, phase)[2]
        curves.append(values)
    data = np.column_stack((times, *curves))
    data.setflags(write=False)
    return data


def plot_transfer_fidelity(ax, mode, source="hyperfine", save=False):
    data = transfer_fidelity_data(mode, source)
    labels = (("g → h0", "e → h1", "+ → (h0+h1)/√2") if mode == "write" else
              (("h0 → g", "h1 → e", "+ → (g+e)/√2") if source == "hyperfine" else
               ("r0 → g", "r1 → e", "+ → (g+e)/√2")))
    for column, (label, style) in enumerate(zip(labels, ("-", "--", "-.")), start=1):
        ax.plot(data[:, 0], data[:, column], style,
                label=f"{label}: final {data[-1, column]:.4f}")
    ax.set(xlabel="Transfer time (µs)", ylabel="Joint state-transfer fidelity",
           title="Transmon → atom" if mode == "write" else "Atom → transmon",
           ylim=(-.02, 1.02), xlim=(data[0, 0], data[-1, 0]))
    ax.legend(fontsize=8, loc="best")
    if save:
        output = Path(__file__).with_name(f"hybrid_transfer_{mode}_fidelity.csv")
        np.savetxt(output, data, delimiter=",", comments="",
                   header="transfer_time_us,ground_joint_fidelity,excited_joint_fidelity,plus_joint_fidelity")
        print(f"Saved {output}", flush=True)


@lru_cache(maxsize=1)
def storage_fidelity_data():
    """Atomic storage and round-trip return, both unconditional joint overlaps.

    Columns: hold time, round-trip noisy/ideal, direct transmon, atomic
    noisy/ideal, excited-input atomic noisy/ideal. Atomic fidelity is measured
    BEFORE readout, against |g,0> tensor (alpha|h0> + beta|h1>). It includes
    write errors. Round-trip return alone also accepts a transmon bypass.
    Separate noiseless write/read phase calibrations are fixed across inputs,
    noise settings and durations. The full parked state is evolved throughout.
    """
    durations = np.unique(np.r_[np.linspace(0, max(2 * STORAGE, 1.0), 41), STORAGE])
    inputs = (g, e, (g + e).unit(), (g - e).unit(),
              (g + 1j * e).unit(), (g - 1j * e).unit())
    phase = read_phase("hyperfine", roundtrip=True)
    write_phase = calibrated_phase()
    curves, atomic_curves, excited_curves = [], [], []
    for noisy in (True, False):
        print(f"Storage fidelity: {'noisy' if noisy else 'ideal'} six-state sweep…",
              flush=True)
        # All drives vanish after MAP; the remaining Hamiltonian is constant.
        # Diagonalizing once avoids resolving fast parked detunings at each hold.
        parked = H(MAP[1])
        solver = qt.MESolver(parked, C_OPS if noisy else [], options={
            "method": "diag", "normalize_output": False,
        })
        values, atomic_values = [], []
        for index, target in enumerate(inputs, start=1):
            print(f"  input {index}/6", flush=True)
            _, written = simulate(target[0, 0], target[1, 0], noisy=noisy,
                                  include_storage=False)
            initial = qt.ket2dm(written[-1]) if written[-1].isket else written[-1]
            effect = retrieval_effect(target, phase, noisy=noisy)
            held = solver.run(initial, durations)
            values.append(np.real(qt.expect(effect, held.states)))
            atomic_values.append(np.real(qt.expect(
                atomic_storage_effect(target, write_phase), held.states)))
        curves.append(np.mean(values, axis=0))
        atomic_curves.append(np.mean(atomic_values, axis=0))
        excited_curves.append(atomic_values[1])

    # Isolated parked transmon, same T1/Tphi and total write+hold+read time.
    # Work in its known rotating frame, removing deterministic precession only.
    read_duration = retrieval_model("hyperfine")[1][-1]
    elapsed = MAP[1] + durations + read_duration
    direct = direct_transmon_fidelity(elapsed)
    data = np.column_stack((durations, curves[0], curves[1], direct,
                            atomic_curves[0], atomic_curves[1],
                            excited_curves[0], excited_curves[1]))
    if (not np.all(np.isfinite(data)) or np.any(data[:, 1:] < -1e-6)
            or np.any(data[:, 1:] > 1 + 1e-6)):
        raise ValueError("Storage fidelity sweep produced invalid probabilities.")
    data.setflags(write=False)
    return data


def plot_storage_fidelity(ax, save=False):
    data = storage_fidelity_data()
    for column, label, style, color in (
            (4, "Atomic six-state mean (noise)", "-", "tab:blue"),
            (5, "Atomic six-state mean (ideal)", "--", "tab:orange"),
            (6, "Atomic excited-input transfer (noise)", "-", "tab:red"),
            (1, "Transmon return (noise)", "-.", "gray"),
            (3, "Direct transmon (noise)", ":", "tab:green")):
        ax.plot(data[:, 0], data[:, column], style, label=label, color=color)
    ax.axvline(STORAGE, color="gray", alpha=.4, linewidth=1)
    ax.set(xlabel="Hold time after writing (µs)", ylabel="Joint fidelity",
           title="Memory transfer and storage", ylim=(-.02, 1.02),
           xlim=(data[0, 0], data[-1, 0]))
    ax.legend(fontsize=8, loc="upper left")
    ax.text(.98, .52, f"Write e → g,0,h1 (noise): {data[0, 6]:.3%}\n"
            "Unchanged-transmon bypass: atomic mean = 1/3.\n"
            "Transmon return can be high without transfer.\n"
            "Fixed phases; no postselection.", transform=ax.transAxes,
            ha="right", va="center", fontsize=7)
    if save:
        output = Path(__file__).with_name("hybrid_storage_fidelity.csv")
        np.savetxt(output, data, delimiter=",", comments="",
                   header="storage_duration_us,noisy_mean_joint_fidelity,ideal_mean_joint_fidelity,direct_transmon_mean_fidelity,atomic_noisy_mean_joint_fidelity,atomic_ideal_mean_joint_fidelity,atomic_noisy_excited_input_fidelity,atomic_ideal_excited_input_fidelity")
        print(f"Saved {output}", flush=True)


def retrieval_main(source="hyperfine", roundtrip=False, check=False, save=False):
    work=progress.transfer_stages(_SETTINGS)
    progress.update(0, "Calibrating read phase")
    if roundtrip and source != "hyperfine":
        raise ValueError("Round trip retrieves from hyperfine memory.")
    phase = read_phase(source, roundtrip)
    progress.update(work["calibration"], "Calculating storage and read dynamics")
    theta = np.deg2rad(parameter("INPUT_THETA", 90.0))
    phi = np.deg2rad(parameter("INPUT_PHI", 60.0))
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    target = (alpha * g + beta * e).unit()
    write_ts = write_states = None
    atomic_joint = None
    if roundtrip:
        # Continue the COMPLETE joint state. Do not reset, postselect,
        # or remove residual cavity photons/entanglement after storage.
        write_ts, write_states = simulate(alpha, beta, noisy=True, include_storage=True)
        initial = write_states[-1]
        atomic_joint = float(np.real(qt.expect(
            atomic_storage_effect(target, calibrated_phase()), initial)))
    else:
        initial = stored_retrieval_initial(alpha, beta, source)
    ts, states, stages, flux_read, voltage_read = evolve_read(initial, source=source)
    raw, corrected, joint, coherence = read_metrics(states, target, phase)

    for state in states:
        rho = qt.ket2dm(state) if state.isket else state
        assert abs(rho.tr() - 1) < 1e-6
    rho_final = qt.ket2dm(states[-1]) if states[-1].isket else states[-1]
    assert (rho_final - rho_final.dag()).norm() < 1e-6
    assert rho_final.eigenenergies().min() > -1e-6
    assert -1e-6 <= corrected[-1] <= 1 + 1e-6

    print("Mode:", "write-store-read round trip" if roundtrip else f"{source} to transmon")
    print(f"Storage hold before retrieval: {STORAGE:g} us")
    if roundtrip:
        print(f"Atomic storage joint fidelity BEFORE retrieval: {atomic_joint:.6f}")
        print("Transmon return overlaps below include excitation that never entered the memory.")
    print(f"Raw final transmon overlap: {raw[-1]:.6f}")
    print(f"Calibrated final transmon overlap: {corrected[-1]:.6f}")
    print(f"Joint overlap (target transmon, cavity vacuum, atom h0): {joint[-1]:.6f}")
    print(f"Final mean cavity photons: {qt.expect(n, states[-1]).real:.6e}")
    print(f"Final atomic h0 population: {qt.expect(Ph0, states[-1]).real:.6f}")
    print(f"Fixed calibrated transmon phase (rad): {phase:.6f}")
    print("Density-matrix checks passed.")

    if check:
        # Test a distinct relative phase with the SAME calibration.
        initial_test = stored_retrieval_initial(1, 1j, source, noisy=False)
        _, test_states, _, _, _ = evolve_read(initial_test, noisy=False, source=source)
        _, refined_states, _, _, _ = evolve_read(
            initial_test, noisy=False, source=source, step=MAX_STEP / 2
        )
        # This isolated-source test uses isolated-source calibration.
        test_phase = read_phase(source) if roundtrip else phase
        test_target = (g + 1j * e).unit()
        value = read_metrics(test_states[-1:], test_target, test_phase)[1][-1]
        refined = read_metrics(refined_states[-1:], test_target, test_phase)[1][-1]
        assert abs(value - refined) < 2e-5
        assert value > 0.5, "Retrieval does not preserve this test superposition."
        print(f"Ideal |+i> retrieval overlap: {value:.6f}")
        print(f"Step refinement overlap change: {abs(value-refined):.3e}")
        print("Phase-sensitive retrieval and step-refinement checks passed.")

    # The round trip uses only the complete state after writing and storage.
    cases = [("After writing and storage" if roundtrip else
              f"Atomic input after {STORAGE:g} µs storage", ts, states, corrected, joint)]
    with progress.scope(work["storage"], work["basis"], "Read logical inputs"):
        labels, probabilities, input_fidelities = basis_probabilities("read", roundtrip, source, phase=phase)
    nominal_pops = np.real([qt.expect(op, states) for op in
                            (Pe, n, Ph0, Ph1, Pr0, Pr1, Plost, Pwrong)])
    with progress.scope(work["basis"], .95, "Read tolerance scenarios"):
        bands = (tolerance_bands.compute(sys.modules[__name__], _SETTINGS, "read", alpha, beta,
                                         phase, ts, nominal_pops, joint, joint[-1], source,
                                         nominal_state=rho_final, nominal_basis=input_fidelities)
                 if not roundtrip else None)
    progress.update(.95, "Building read figure")
    summary_row = 2
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(summary_row + 1, 2)
    flux_axis = fig.add_subplot(grid[0, 0])
    voltage_axis = flux_axis.twinx()
    detuning_axis = fig.add_subplot(grid[0, 1])
    performance_axis = fig.add_subplot(grid[summary_row, 0])
    flux_line, = flux_axis.plot(ts, [flux_read(t) for t in ts],
                               color="tab:blue", label="Flux (left axis)", zorder=3)
    voltage_line, = voltage_axis.plot(ts, [voltage_read(t) for t in ts],
                                     color="tab:orange", linestyle="--",
                                     label="Voltage (right axis)", zorder=2)
    flux_axis.set_ylabel("Flux / flux quantum", color="tab:blue")
    voltage_axis.set_ylabel("Electrode voltage (V)", color="tab:orange")
    flux_axis.tick_params(axis="y", colors="tab:blue")
    voltage_axis.tick_params(axis="y", colors="tab:orange")
    flux_axis.spines["left"].set_color("tab:blue")
    voltage_axis.spines["right"].set_color("tab:orange")
    flux_axis.set_title("Flux and voltage control")
    voltage_axis.legend(handles=[flux_line, voltage_line], loc="upper center",
                        bbox_to_anchor=(0.5, 0.88), ncol=2, fontsize=8,
                        framealpha=1)
    fields = np.asarray([voltage_read(t) / GAP_CM + STRAY_FIELD for t in ts])
    transmon_detuning = np.asarray([transmon_frequency(flux_read(t)) - FC for t in ts])
    atom_detuning = F_A_ZERO - FC + (K_R1 - K_R0) * fields**2
    detuning_axis.plot(ts, transmon_detuning, color="tab:blue", label="Transmon")
    detuning_axis.plot(ts, atom_detuning, color="tab:orange", linestyle="--", label="Atom")
    detuning_axis.axhline(0, color="gray", linewidth=.7, alpha=.5)
    detuning_axis.set(title="Detuning from cavity", ylabel="Detuning (MHz)")
    detuning_axis.legend(fontsize=8)
    if roundtrip:
        plot_storage_fidelity(performance_axis, save=save)
    if not roundtrip:
        pair = ("h0", "h1") if source == "hyperfine" else ("r0", "r1")
        final_populations = [float(np.real(qt.expect(op, rho_final)))
                             for op in (n, Ph0, Pr0 + Pr1, Plost, Pwrong)]
        summary_rows = [
            ["Storage hold before readout", f"{STORAGE:g} µs"],
            ["Readout duration (including optical operations)", f"{ts[-1] * 1000:.2f} ns"],
            ["Total storage + readout duration", f"{STORAGE + ts[-1]:.6g} µs"],
            ["Atom–cavity rounded pulse", f"{(AC[1] - AC[0]) * 1000:.2f} ns"],
            ["Cavity–transmon rounded pulse", f"{(TC[1] - TC[0]) * 1000:.2f} ns"],
            [f"Joint fidelity: {pair[0]} → g", f"{input_fidelities[0]:.2%}"],
            [f"Joint fidelity: {pair[1]} → e", f"{input_fidelities[1]:.2%}"],
            ["Joint fidelity: + → +", f"{input_fidelities[2]:.2%}"],
            ["Selected input: joint fidelity", f"{joint[-1]:.2%}"],
            ["Selected input: transmon coherence 2|ρ(g,e)|", f"{coherence[-1]:.6f}"],
            ["Selected input: residual mean cavity photons", f"{final_populations[0]:.3e}"],
            ["Selected input: atom reset h0 population", f"{final_populations[1]:.6f}"],
            ["Selected input: residual r0 + r1 population", f"{final_populations[2]:.3e}"],
            ["Selected input: atomic loss population", f"{final_populations[3]:.3e}"],
            ["Selected input: wrong-Rydberg population", f"{final_populations[4]:.3e}"],
        ]
        summary_rows = tolerance_bands.table_rows(summary_rows, bands, "read")
        performance_axis.set_title("Storage + readout summary")
        table = results_table(performance_axis, ["Metric", "Nominal", "With tolerances"], summary_rows,
                              widths=[.58, .16, .26], bbox=(0, 0.04, 1, 0.94), fontsize=7.5)
        for row in range(1, len(summary_rows) + 1):
            table[row, 0].set_text_props(ha="left")
        if save:
            data = transfer_fidelity_data("read", source)
            csv_output = Path(__file__).with_name("hybrid_transfer_read_fidelity.csv")
            np.savetxt(csv_output, data, delimiter=",", comments="",
                       header="transfer_time_us,ground_joint_fidelity,excited_joint_fidelity,plus_joint_fidelity")
            print(f"Saved {csv_output}", flush=True)
    for index, (label, _, _, _, _) in enumerate(cases):
        basis_axis = fig.add_subplot(grid[summary_row, index + 1], projection="3d")
        selection = slice(3 * index, 3 * index + 3)
        title = "Transmon return after write/store" if roundtrip else "After storage + retrieval: " + source
        title += " (nominal)" if bands else ""
        plot_basis_columns(basis_axis, labels[selection], probabilities[selection], title,
                           mode="read", input_label="Original transmon input" if roundtrip else None)
        if not roundtrip:
            basis_axis.set_box_aspect((4, 4, 3), zoom=.98)
            basis_axis.xaxis.labelpad = -5
    summary_lines = [f"{label}: selected-input overlap = {case_corrected[-1]:.6f}; "
                     f"joint overlap = {case_joint[-1]:.6f}"
                     for label, _, _, case_corrected, case_joint in cases]
    if roundtrip:
        summary_lines.insert(0, f"Atomic storage BEFORE retrieval: selected-input joint fidelity = {atomic_joint:.6f}.")
        summary_lines.append("Fixed phase correction; basis columns use joint targets |g,0,h0⟩ / |e,0,h0⟩.")
        summary_lines.append(f"Round-trip return fidelities: input 0={input_fidelities[0]:.4f}, "
                             f"input 1={input_fidelities[1]:.4f}, +={input_fidelities[2]:.4f}.")
        summary_lines.append("Return fidelity includes states that remained in the transmon; see atomic storage curves for transfer.")
        summary_lines.append("+ = equal coherent superposition; olive: intended logical outputs; Other: remaining joint states.")
    else:
        lower, upper = ("h0", "h1") if source == "hyperfine" else ("r0", "r1")
        summary_lines = [
            f"Atomic input: θ={np.rad2deg(theta):g}°, φ={np.rad2deg(phi):g}°.",
            f"Initial atomic populations (before storage): {lower} = {100 * abs(alpha)**2:.2f}%, "
            f"{upper} = {100 * abs(beta)**2:.2f}%.",
        ]
    fig.supxlabel("\n".join(summary_lines), fontsize=9)
    population_axes = []
    time_axes = [flux_axis, detuning_axis]
    for column, (label, case_ts, case_states, _, _) in enumerate(cases):
        populations = fig.add_subplot(grid[1, column] if len(cases) > 1 else grid[1, :])
        population_axes.append(populations)
        time_axes.append(populations)
        for op, name in ((Pe, "Transmon e"), (n, "Cavity photons"),
                         (Pr0, "r0"), (Pr1, "r1"), (Ph0, "h0"),
                         (Ph1, "h1"), (Pwrong, "Wrong Rydberg"), (Plost, "Loss")):
            populations.plot(case_ts, np.real(qt.expect(op, case_states)), label=name)
        populations.set(ylabel="Population / photon number", title=label)
        populations.legend(fontsize=8, ncol=2)
    if bands:
        tolerance_bands.draw(bands, flux_axis, detuning_axis, population_axes[0], read=True)
    # Match population scales without assuming photon number is bounded by one.
    upper = max(ax.get_ylim()[1] for ax in population_axes)
    for ax in population_axes:
        ax.set_ylim(-0.03, max(1.03, upper))
    for ax in time_axes:
        ax.set(xlabel="Retrieval time (us)", xlim=(ts[0], ts[-1]))
        for _, (start, end) in stages:
            ax.axvline(start, color="gray", alpha=0.15)
            ax.axvline(end, color="gray", alpha=0.15)
    title = "Write–store–read: transmon return" if roundtrip else "Atom to transmon"
    subtitle = "After writing and storage" if roundtrip else f"Retrieval from {source} memory after {STORAGE:g} µs storage"
    fig.suptitle(title + "\n" + subtitle + (" — shaded sampled tolerance ranges" if bands else ""))
    fig.canvas.manager.set_window_title(title)
    if _ASSISTANT_OUTPUT:
        names = ("transmon_e", "cavity_mean_photons", "atom_h0", "atom_h1",
                 "atom_r0", "atom_r1", "atom_loss", "atom_wrong_rydberg")
        output_pops = np.asarray([np.real(qt.expect(op, states))
                                 for op in (Pe, n, Ph0, Ph1, Pr0, Pr1, Plost, Pwrong)])
        write_report(_ASSISTANT_OUTPUT, "roundtrip" if roundtrip else "read", _SETTINGS,
            metadata={"noise_enabled": True, "source": source, "roundtrip": roundtrip,
                      "input_theta_degrees": float(np.rad2deg(theta)), "input_phi_degrees": float(np.rad2deg(phi)),
                      "phase_correction_rad": float(phase), "stages_us": dict(stages),
                      "storage_duration_us": STORAGE,
                      "time_origin": "Retrieval begins at zero AFTER storage; initial state already includes the hold.",
                      "units": "time us; detuning MHz; flux Phi/Phi0; voltage V; populations/fidelity/coherence dimensionless; cavity is mean photon number",
                      "fidelity_definition": "Unconditional joint target overlap; fixed noiseless phase; no postselection.",
                      "basis_evaluation": "After storage and retrieval; independent atomic inputs unless roundtrip."},
            summary={"tolerance_ranges": bands["summary"] if bands else None,
                     "displayed_metrics": summary_rows if not roundtrip else summary_lines,
                     "selected_joint_fidelity": float(joint[-1]), "selected_raw_transmon_overlap": float(raw[-1]),
                     "selected_phase_corrected_transmon_overlap": float(corrected[-1]),
                     "selected_transmon_coherence": float(coherence[-1]),
                     "atomic_joint_before_retrieval_if_roundtrip": atomic_joint},
            traces={**tolerance_bands.report_traces(bands),
                    "selected_input_dynamics": trace(ts, joint_fidelity=joint,
                        raw_transmon_overlap=raw, phase_corrected_transmon_overlap=corrected,
                        transmon_coherence_2abs_rho_g_e=coherence, **dict(zip(names, output_pops))),
                    "controls": trace(ts, flux_Phi_over_Phi0=[flux_read(t) for t in ts],
                        electrode_voltage_V=[voltage_read(t) for t in ts],
                        transmon_detuning_MHz=transmon_detuning, atom_detuning_MHz=atom_detuning)},
            basis={"input_labels": labels, "joint_output_labels": ["g,0,h0", "e,0,h0", "wrong_rydberg", "remaining_other"],
                   "probabilities": probabilities, "joint_fidelities": input_fidelities})
    stem = "hybrid_roundtrip" if roundtrip else f"hybrid_retrieval_{source}"
    output = Path(__file__).with_name(stem + ".png")
    if save:
        fig.savefig(output, dpi=170)
        np.savetxt(output.with_suffix(".csv"),
                   np.column_stack((ts, raw, corrected, joint, coherence)),
                   delimiter=",", comments="",
                   header="retrieval_time_us,raw_transmon_overlap,calibrated_transmon_overlap,joint_overlap,transmon_coherence")
        print(f"Saved {output}")


def paper_figure(save=False, show=True):
    # (a,b) One-excitation manifold, transmon parked:
    # |e,0,r0>, |g,1,r0>, |g,0,r1>.
    # Subtract the common r0 energy; keep the off-resonant transmon
    # so its dispersive cavity shift is present.
    span = 3.0 / abs(K_R1-K_R0)
    fields = np.sqrt(np.linspace(max(0.0,E_SWAP**2-span),E_SWAP**2+span,401))
    detunings = F_A_ZERO-FC+(K_R1-K_R0)*fields**2
    volts = GAP_CM * (fields - STRAY_FIELD)
    eigenvalues, atom_weights = [], []
    dt = F_T_PARK - FC
    for da in detunings:
        matrix = qt.Qobj([[dt, G_TC/TWOPI, 0],
                          [G_TC/TWOPI, 0, G_AC/TWOPI],
                          [0, G_AC/TWOPI, da]])
        energies, vectors = matrix.eigenstates()
        eigenvalues.append(energies[:2])
        atom_weights.append([abs(v.full()[2, 0])**2 for v in vectors[:2]])
    eigenvalues, atom_weights = np.asarray(eigenvalues), np.asarray(atom_weights)
    gap_index = np.argmin(eigenvalues[:,1] - eigenvalues[:,0])
    v_res = volts[gap_index]

    # (c) Full-system exchange at constant voltages; no optical drive.
    # Initial |g,1,r0>. Includes the same transmon, cavity and atomic losses.
    ts = np.linspace(0, 1.5, 601)
    psi = qt.tensor(g, qt.basis(NC, 1), r0)
    populations = []
    for field in (E_SWAP, E_PARK):
        H = (H_coupling + TWOPI * dt * Pe
             + TWOPI * K_R0 * field**2 * Pr0
             + TWOPI * (F_A_ZERO-FC+K_R1*field**2) * Pr1
             + TWOPI * (F_WRONG_SPLIT+K_WRONG*field**2) * Pwrong)
        result = qt.mesolve(
            H, qt.ket2dm(psi), ts, c_ops=C_OPS, e_ops=[Pr1],
            options={"method":"dop853", "atol":1e-11, "rtol":1e-9,
                     "max_step":MAX_STEP, "nsteps":100000},
        )
        populations.append(np.real(result.expect[0]))
    populations = np.asarray(populations)

    # (d) Isolate optical leakage; no phenomenological failure probability.
    durations = np.geomspace(0.0002, T_OPT, 45)
    prep = np.asarray([isolated_preparation(d)[:, -1] for d in durations])

    # (e) Retrieval population table for its two valid basis inputs:
    # transmon initially g and cavity empty, atom h0 or h1.
    # Exclusive outputs: |g,0,h0>, |e,0,h0>, and everything else.
    # This is not a CNOT truth table and does not characterize phase fidelity.
    outputs = [qt.tensor(t, qt.basis(NC,0), h0) for t in (g,e)]
    table = []
    for alpha, beta in ((1,0),(0,1)):
        initial = retrieval_initial(alpha,beta)
        _, states, _, _, _ = evolve_read(initial, noisy=True)
        rho = states[-1]
        probs = [float(np.real(qt.expect(qt.ket2dm(k),rho))) for k in outputs]
        table.append(probs + [1-sum(probs)])
    table = np.asarray(table)
    assert np.all(table >= -1e-7)
    assert np.allclose(table.sum(axis=1),1,atol=1e-7)
    assert np.all((populations >= -1e-7) & (populations <= 1+1e-7))
    # Quantitative check of the diagonalization and normalization.
    assert np.all(np.diff(eigenvalues,axis=1) > 0)
    assert np.all((atom_weights>=0) & (atom_weights<=1+1e-12))
    print("Retrieval population table (g/0/h0, e/0/h0, other):")
    print(table)
    print(f"Minimum avoided-crossing gap: "
          f"{(eigenvalues[gap_index,1]-eigenvalues[gap_index,0]):.6f} MHz")
    print("Figure probability and spectrum checks passed.")

    style = {"font.family":"DejaVu Serif","font.size":10,
             "axes.labelsize":11,"axes.titlesize":11,
             "xtick.direction":"in","ytick.direction":"in",
             "xtick.top":True,"ytick.right":True}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(15,9),layout="constrained")
        gs = fig.add_gridspec(2,3)
        axes = [fig.add_subplot(gs[0,j]) for j in range(3)]
        axes.append(fig.add_subplot(gs[1,0]))
        axes.append(fig.add_subplot(gs[1,1],projection="3d"))
        axes.append(fig.add_subplot(gs[1,2]))
        blue, dark = "#3346c8","#303030"
        # Voltage shown in mV for readable offsets near the crossing.
        x = volts*1000
        ax=axes[0]
        ax.plot(x,eigenvalues[:,0],color=dark,label="Lower dressed branch")
        ax.plot(x,eigenvalues[:,1],color=blue,ls="--",label="Upper dressed branch")
        ax.axvline(v_res*1000,color="gray",lw=.8,ls=":")
        ax.set(xlabel="Electrode bias (mV)",ylabel="Relative energy / h (MHz)",
               title="Dressed energy spectrum")
        ax.legend(fontsize=8,loc="upper left")

        ax=axes[1]
        ax.plot(x,atom_weights[:,0],color=dark,label="Lower branch")
        ax.plot(x,atom_weights[:,1],color=blue,ls="--",label="Upper branch")
        ax.axvline(v_res*1000,color="gray",lw=.8,ls=":")
        ax.set(xlabel="Electrode bias (mV)",
               ylabel=r"Atomic excitation weight $|\langle g,0,r_1|\psi_k\rangle|^2$",
               title="Atom-photon hybridization",ylim=(-.03,1.03))
        ax.legend(fontsize=8)

        ax=axes[2]
        ax.plot(ts,populations[0],color=blue,ls="--",
                label="Configured swap bias")
        ax.plot(ts,populations[1],color=dark,label="Atom parked / detuned")
        ax.axvline(np.pi/(2*G_AC),color="gray",lw=.8,ls=":")
        ax.set(xlabel="Interaction time (us)",ylabel=r"$P(r_1)$",
               title="Cavity excitation transferred to atom",ylim=(-.03,1.03))
        ax.legend(fontsize=8,loc="upper right")

        ax=axes[3]
        ax.semilogx(durations*1000,prep[:,1],color=blue,label="Desired Rydberg")
        ax.semilogx(durations*1000,prep[:,2],color=blue,ls="--",label="Wrong Rydberg")
        ax.semilogx(durations*1000,prep[:,0],color=dark,ls=":",label="Ground state")
        ax.set(xlabel="Preparation pulse duration (ns)",ylabel="Final population",
               title="Optical preparation and leakage",ylim=(-.03,1.03))
        ax.legend(fontsize=8,loc="center right")
        ax.text(.97,.06,"Equal-area smooth pulses\nNeighbor: 1.32 GHz; dipole ratio: 1",
                transform=ax.transAxes,ha="right",fontsize=8)

        ax=axes[4]
        for i in range(2):
            for j in range(3):
                color = "#c3ce48" if j==i else "#c89b32"
                ax.bar3d(i-.23,j-.23,0,.46,.46,max(table[i,j],0),
                         color=color,edgecolor="#4b4934",linewidth=.5,
                         shade=True,alpha=.93)
        ax.set_xticks([0,1],["h0","h1"])
        ax.set_yticks([0,1,2],["g,0,h0","e,0,h0","Other"])
        ax.set(xlabel="Atomic input",ylabel="Joint output",zlabel="Probability",
               zlim=(0,1),title="Retrieval basis populations")
        ax.view_init(elev=25,azim=-53)
        ax.tick_params(labelsize=8)
        ax.text2D(.00,.92,"Input transmon g; cavity vacuum\nOther: all remaining joint states",
                  transform=ax.transAxes,fontsize=8)

        ax=axes[5]
        ax.set_axis_off()
        ax.set_title("State-transfer protocol",pad=12)
        xx=[.10,.36,.62,.88]
        for yy,names,labels in [
            (.76,["T","C","R","H"],["Flux","Bias voltage","Lasers"]),
            (.36,["T","C","R","H"],["Flux","Bias voltage","Lasers"])]:
            for xpos,name in zip(xx,names):
                box=FancyBboxPatch((xpos-.065,yy-.065),.13,.13,
                    boxstyle="round,pad=0.01",transform=ax.transAxes,
                    facecolor="#f4f2ed",edgecolor=dark)
                ax.add_patch(box)
                ax.text(xpos,yy,name,transform=ax.transAxes,ha="center",va="center",
                        fontsize=13)
            for j,label in enumerate(labels):
                destination, origin = (
                    ((xx[j]+.08,yy),(xx[j+1]-.08,yy)) if yy < .5
                    else ((xx[j+1]-.08,yy),(xx[j]+.08,yy))
                )
                ax.annotate("",xy=destination,xytext=origin,
                            xycoords="axes fraction",
                            arrowprops={"arrowstyle":"->","color":dark})
                ax.text((xx[j]+xx[j+1])/2,yy+.105,label,
                        transform=ax.transAxes,ha="center",fontsize=8)
        ax.text(.02,.98,"Write",transform=ax.transAxes,fontsize=11)
        ax.text(.02,.56,"Read",transform=ax.transAxes,fontsize=11)
        ax.text(.5,.05,"T: transmon   C: cavity\nR: Rydberg pair   H: hyperfine memory",
                transform=ax.transAxes,ha="center",fontsize=9)
        ax.annotate("Storage",xy=(.88,.47),xytext=(.88,.61),
                    xycoords="axes fraction",ha="center",fontsize=9,
                    arrowprops={"arrowstyle":"->","color":dark})

        for letter,ax in zip("abcdef",axes):
            if letter=="e":
                ax.text2D(-.04,1.02,letter,transform=ax.transAxes,
                          fontsize=15,fontweight="bold")
            else:
                ax.text(-.10,1.04,letter,transform=ax.transAxes,
                        fontsize=15,fontweight="bold")
        fig.suptitle("Hybrid transmon-cavity-atom quantum memory",
                     fontsize=17)
        fig.supxlabel(
            "Illustrative parameters; fixed cavity and flux-tunable transmon. "
            "Dressed crossing includes the parked transmon. "
            "Population tables do not measure phase fidelity.",
            fontsize=9)
        output=Path(__file__).with_name("hybrid_paper_figure")
        if save:
            fig.savefig(output.with_suffix(".png"),dpi=200)
            fig.savefig(output.with_suffix(".svg"))
    if save:
        np.savez(output.with_suffix(".npz"),voltage_V=volts,
                 energy_MHz=eigenvalues,atom_weight=atom_weights,
                 exchange_time_us=ts,exchange_probabilities=populations,
                 prep_duration_us=durations,prep_final_populations=prep,
                 retrieval_basis_probabilities=table)
        print(f"Saved {output.with_suffix('.png')}")
    if show:
        plt.show(block=True)

def write_paper_figure(save=False, show=True):
    """Paper-style summary for transmon -> hyperfine memory.
    Two valid logical inputs: transmon g/e, cavity vacuum, atom h0.
    Joint outputs are exclusive; 'Other' includes residual excitation/loss.
    """
    durations = np.geomspace(0.0002, T_OPT, 45)
    preparation = [isolated_preparation(d) for d in durations]
    prep = np.asarray([populations[:, -1] for populations in preparation])
    wrong_peak = np.asarray([populations[2].max() for populations in preparation])
    joint_outputs = [
        qt.tensor(g, qt.basis(NC, 0), h0),
        qt.tensor(g, qt.basis(NC, 0), h1),
    ]
    table, traces = [], []
    for alpha, beta in ((1, 0), (0, 1)):
        ts, states = simulate(alpha, beta, noisy=True, include_storage=False)
        final = states[-1]
        probabilities = [
            float(np.real(qt.expect(qt.ket2dm(k), final)))
            for k in joint_outputs
        ]
        table.append(probabilities + [1 - sum(probabilities)])
        traces.append({
            "h0": np.real(qt.expect(Ph0, states)),
            "h1": np.real(qt.expect(Ph1, states)),
            "r1": np.real(qt.expect(Pr1, states)),
            "photon": np.real(qt.expect(n, states)),
        })
        assert max(abs(state.tr() - 1) for state in states) < 1e-6
        assert final.eigenenergies().min() > -1e-6
    table = np.asarray(table)
    assert np.all(table >= -1e-7)
    assert np.allclose(table.sum(axis=1), 1, atol=1e-7)
    print("Write population table (g/0/h0, g/0/h1, other):", flush=True)
    print(table, flush=True)

    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.labelsize": 11, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    }):
        fig = plt.figure(figsize=(13, 9), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(grid[0, 0])
        ax_leak = fig.add_subplot(grid[0, 1])
        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = fig.add_subplot(grid[1, 1], projection="3d")
        blue, black = "#3346c8", "#303030"
        ax0.semilogx(durations * 1000, prep[:, 1], color=blue,
                     label="Desired Rydberg")
        ax0.semilogx(durations * 1000, prep[:, 2], color=blue, ls="--",
                     label="Wrong Rydberg")
        ax0.semilogx(durations * 1000, prep[:, 0], color=black, ls=":",
                     label="Initial hyperfine state")
        ax0.set(xlabel="Preparation pulse duration (ns)",
                ylabel="Final population", ylim=(-.03, 1.03),
                title="Rydberg preparation")
        ax0.legend(fontsize=8, loc="center right")
        ax0.text(.97, .05,
                 f"Neighbor splitting: {F_WRONG_SPLIT/1000:.2f} GHz\n"
                 f"Relative dipole amplitude: {REL_WRONG_DIPOLE:g}",
                 transform=ax0.transAxes, ha="right", fontsize=8)

        # A separate log panel resolves leakage too small for the linear plot.
        ax_leak.loglog(durations * 1000, np.maximum(prep[:, 2], 1e-18),
                       color=blue, ls="--", label="Wrong Rydberg at pulse end")
        ax_leak.loglog(durations * 1000, np.maximum(wrong_peak, 1e-18),
                       color="#b37c27", label="Peak wrong Rydberg (sampled)")
        ax_leak.set(xlabel="Preparation pulse duration (ns)", ylabel="Population",
                    title="Transient and final preparation leakage")
        ax_leak.legend(fontsize=8)

        ax1.plot(ts, traces[1]["photon"], color="#b37c27", ls=":",
                 label="Cavity photons | input e")
        ax1.plot(ts, traces[1]["r1"], color=blue, ls="--",
                 label="P(r1) | input e")
        ax1.plot(ts, traces[1]["h1"], color=blue,
                 label="P(h1) | input e")
        ax1.plot(ts, traces[0]["h0"], color=black,
                 label="P(h0) | input g")
        for interval in (PREP, TC, AC, MAP):
            for boundary in interval:
                ax1.axvline(boundary, color="gray", alpha=.17, lw=.7)
        ax1.set(xlabel="Write-sequence time (us)", ylabel="Population / photon number",
                ylim=(-.03, 1.03), title="Transmon to atom: transfer dynamics")
        ax1.legend(fontsize=7.5, loc="center left")

        for i in range(2):
            for j in range(3):
                ax2.bar3d(i-.23, j-.23, 0, .46, .46, max(table[i,j], 0),
                          color="#c3ce48" if j==i else "#c89b32",
                          edgecolor="#4b4934", linewidth=.5, shade=True, alpha=.93)
        ax2.set_xticks([0,1], ["g", "e"])
        ax2.set_yticks([0,1,2], ["g,0,h0", "g,0,h1", "Other"])
        ax2.set(xlabel="Transmon input", ylabel="Joint output",
                zlabel="Probability", zlim=(0,1), title="Write basis populations")
        ax2.tick_params(labelsize=8)
        ax2.view_init(elev=25, azim=-53)
        ax2.text2D(.0, .91, "Input atom h0; cavity vacuum\nOther: all remaining joint states",
                   transform=ax2.transAxes, fontsize=8)
        ax_leak.text(-.11, 1.04, "b", transform=ax_leak.transAxes, fontsize=15, fontweight="bold")
        ax0.text(-.11, 1.04, "a", transform=ax0.transAxes, fontsize=15, fontweight="bold")
        ax1.text(-.11, 1.04, "c", transform=ax1.transAxes, fontsize=15, fontweight="bold")
        ax2.text2D(-.05, 1.04, "d", transform=ax2.transAxes, fontsize=15, fontweight="bold")
        fig.suptitle("Quantum-state transfer: transmon to atomic memory", fontsize=16)
        fig.supxlabel("Illustrative cavity-transfer model; parameters and timing are unoptimized. "
                      "Basis populations do not characterize phase fidelity.", fontsize=9)

    if save:
        output = Path(__file__).with_name("hybrid_write_paper_figure")
        fig.savefig(output.with_suffix(".png"), dpi=200)
        fig.savefig(output.with_suffix(".svg"))
        np.savez(output.with_suffix(".npz"),
                 preparation_duration_us=durations,
                 preparation_final_populations=prep,
                 preparation_wrong_peak=wrong_peak,
                 write_time_us=ts, write_basis_probabilities=table,
                 cavity_photons_input_e=traces[1]["photon"],
                 rydberg_r1_input_e=traces[1]["r1"],
                 memory_h1_input_e=traces[1]["h1"],
                 memory_h0_input_g=traces[0]["h0"])
        print(f"Saved {output.with_suffix('.png')}", flush=True)
    if show:
        plt.show(block=True)


def simultaneous_main(save=False):
    """Write to memory with both transitions resonant with the cavity together."""
    from simultaneous_coupling import run
    return run(sys.modules[__name__], save=save)


def all_figures(save=False):
    """Build the two compact windows before a single blocking show call."""
    jobs = [
        ("Transmon to atom", lambda: main(save=save)),
        ("Atom to transmon", lambda: retrieval_main(save=save)),
    ]
    for index, (name, build) in enumerate(jobs, start=1):
        print(f"Generating figure {index}/{len(jobs)}: {name}", flush=True)
        with progress.scope((index-1)/len(jobs), index/len(jobs), name):
            build()
    count = len(plt.get_fignums())
    print(f"All {count} figures are ready. Close their windows to finish.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from hybrid_panel import launch
        launch(backend_path=Path(__file__).resolve())
        raise SystemExit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", action="store_true", help="Emit structured panel progress.")
    parser.add_argument("--parameter-sweep", action="store_true", help="Sweep one model input and compare write/read fidelities.")
    parser.add_argument("--sweep-parameter", default="PHI_SWAP")
    parser.add_argument("--sweep-start", type=float, default=0.26)
    parser.add_argument("--sweep-stop", type=float, default=0.27)
    parser.add_argument("--sweep-step", type=float, default=0.001)
    parser.add_argument("--sweep-direction", choices=("both", "write", "read"), default="both")
    parser.add_argument("--sweep-point", help=argparse.SUPPRESS)
    parser.add_argument("--simultaneous", action="store_true", help="Write with simultaneous transmon/atom/cavity resonance.")
    parser.add_argument("--tolerance-scan", action="store_true", help="Scan fixed +/- tolerances and show sampled best/worst fidelity.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--leakage-scan", action="store_true")
    parser.add_argument("--retrieve", action="store_true",
                        help="Store a prepared atomic qubit for STORAGE, then transfer it to the transmon.")
    parser.add_argument("--roundtrip", action="store_true",
                        help="Write, store, then retrieve the same joint state.")
    parser.add_argument("--source", choices=("hyperfine", "rydberg"), default="hyperfine")
    parser.add_argument("--paper-figure", action="store_true", help="Generate a six-panel scientific summary.")
    parser.add_argument("--save", action="store_true",
                        help="Also export plot/data files; disabled by default.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open plot windows (for unattended runs).")
    parser.add_argument("--all-figures", action="store_true",
                        help="Generate independent transmon-to-atom and atom-to-transmon windows (default).")
    parser.add_argument("--write-paper-figure", action="store_true",
                        help="Paper-style preparation, dynamics, and write probabilities.")
    parser.add_argument("--run", action="store_true", help="Run figures directly, bypassing the startup panel.")
    parser.add_argument("--config", help="JSON object of parameter overrides.")
    parser.add_argument("--assistant-output", help="Directory for run-specific numerical assistant reports.")
    parser.add_argument("--write-only", action="store_true", help="Only the transmon-to-atom transfer figure.")
    args = parser.parse_args()
    progress.enable(args.progress or os.environ.get("HYBRID_SIM_PROGRESS")=="1")
    progress.update(0, "Preparing simulation")
    _ASSISTANT_OUTPUT = args.assistant_output
    selected_modes = sum((args.paper_figure, args.retrieve, args.roundtrip,
                          args.leakage_scan, args.all_figures, args.write_paper_figure, args.write_only, args.tolerance_scan, args.simultaneous, args.parameter_sweep, bool(args.sweep_point)))
    if selected_modes > 1:
        parser.error("Choose one figure mode, or use --all-figures.")
    if args.all_figures and args.check:
        parser.error("--check runs verification separately from --all-figures.")
    if not args.simultaneous:
        diagnostics = derived(_SETTINGS)
        print(f"Pulse-peak detuning (MHz): "
              f"transmon-cavity={diagnostics['transmon_swap_detuning_MHz']:.6g}; "
              f"atom-cavity={diagnostics['atom_swap_detuning_MHz']:.6g}", flush=True)
        print(f"{'Manual' if _SETTINGS['DIRECT_COUPLING'] else 'Calculated'} g_TC / 2pi: {G_TC/TWOPI:.6g} MHz; "
              f"g_AC / 2pi: {G_AC/TWOPI:.6g} MHz", flush=True)
    if args.sweep_point:
        from parameter_sweep import evaluate_point, atomic_json
        directions = ("write", "read") if args.sweep_direction == "both" else (args.sweep_direction,)
        atomic_json(args.sweep_point, evaluate_point(sys.modules[__name__], directions))
    elif args.parameter_sweep:
        import parameter_sweep
        try:
            parameter_sweep.run(sys.modules[__name__], _SETTINGS,
                                args.sweep_parameter, args.sweep_start, args.sweep_stop,
                                args.sweep_step, args.sweep_direction, save=args.save,
                                output_directory=_ASSISTANT_OUTPUT)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.simultaneous:
        try:
            simultaneous_main(save=args.save)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.tolerance_scan:
        import tolerance_scan
        tolerance_scan.run(sys.modules[__name__], _SETTINGS, save=args.save,
                           output_directory=_ASSISTANT_OUTPUT)
    elif args.all_figures or (selected_modes == 0 and not args.check):
        all_figures(save=args.save)
    elif args.write_only:
        main(check=args.check, save=args.save)
    elif args.write_paper_figure:
        write_paper_figure(save=args.save, show=False)
    elif args.paper_figure:
        paper_figure(save=args.save, show=False)
    elif args.retrieve or args.roundtrip:
        retrieval_main(args.source, args.roundtrip, args.check, save=args.save)
    elif args.leakage_scan:
        leakage_scan(save=args.save)
    else:
        main(args.check, save=args.save)

    progress.update(1, "Simulation ready", complete=True)
    if not args.no_show and not args.check:
        # Blocking interactive windows: close the windows to finish the run.
        plt.show(block=True)
    else:
        plt.close("all")
