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
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import qutip as qt
from pathlib import Path

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

from hybrid_settings import validate
_SETTINGS = validate(_SETTINGS)

def parameter(name, default):
    return _SETTINGS.get(name, default)

TWOPI = 2 * np.pi

# Device parameters: EJ/h and EC/h, in MHz.
FC = parameter("FC", 5000.0) # cavity resonance frequency
EC = parameter("EC", 250.0) # transmon charging energy
EJ_MAX = parameter("EJ_MAX", 20000.0) # maximum JJ energy
ASYMMETRY = parameter("ASYMMETRY", 0.10) # asymmetry between JJ
F_T_PARK = parameter("F_T_PARK", 5800.0) # transmon frequency before its changing with the cavity
G_TC = TWOPI * parameter('G_TC', 20.0) # transmon to cavity coupling strength
G_AC = TWOPI * parameter('G_AC', 0.5) #atom-cavity coupling strength

# Example low-field Stark coefficients, defined by shift/h = K*E^2.
# K is MHz/(V/cm)^2, already INCLUDING the conventional -alpha/(2h).
# These are placeholders, not parameters of a selected rubidium pair.
F_A_ZERO = parameter("F_A_ZERO", 5150.0)
K_R0 = parameter("K_R0", -2.0)
K_R1 = parameter("K_R1", -12.0)
GAP_CM = parameter("GAP_CM", 0.10)
STRAY_FIELD = parameter("STRAY_FIELD", 0.20)
E_PARK = parameter("E_PARK", 0.0)
E_SWAP = np.sqrt((FC - F_A_ZERO) / (K_R1 - K_R0))

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
OMEGA = TWOPI * parameter('OMEGA', 10.0)
T_OPT = TWOPI / OMEGA  # sin^2 envelope averages to 1/2
RAMP = parameter("RAMP", 0.002)
GAP = parameter("GAP", 0.01)
STORAGE = parameter("STORAGE", 5.0)
MAX_STEP = parameter("MAX_STEP", 0.0005)

def flux_for_frequency(f):
    # Invert the approximate asymmetric-SQUID transmon formula,
    # choosing the branch 0 <= Phi/Phi0 <= 1/2.
    ratio = ((f + EC)**2 / (8 * EC)) / EJ_MAX
    c2 = (ratio**2 - ASYMMETRY**2) / (1 - ASYMMETRY**2)
    if not 0 <= c2 <= 1:
        raise ValueError("Requested transmon frequency is outside tuning range.")
    return np.arccos(np.sqrt(c2)) / np.pi

PHI_PARK = flux_for_frequency(F_T_PARK)
PHI_SWAP = flux_for_frequency(FC)

def transmon_frequency(phi):
    ej = EJ_MAX * np.sqrt(
        np.cos(np.pi * phi)**2
        + ASYMMETRY**2 * np.sin(np.pi * phi)**2
    )
    return np.sqrt(8 * EC * ej) - EC

# Stage boundaries; swap hold times are nominal, not numerically optimized.
PREP = (0.0, T_OPT)
TC = (PREP[1] + GAP, PREP[1] + GAP + 2 * RAMP + np.pi / (2 * G_TC))
AC = (TC[1] + GAP, TC[1] + GAP + 2 * RAMP + np.pi / (2 * G_AC))
MAP = (AC[1] + GAP, AC[1] + GAP + T_OPT)
T_END = MAP[1] + STORAGE

def smooth_window(t, start, end):
    """Raised-cosine ramps with a flat central plateau."""
    if t <= start or t >= end:
        return 0.0
    if t < start + RAMP:
        return 0.5 * (1 - np.cos(np.pi * (t - start) / RAMP))
    if t > end - RAMP:
        return 0.5 * (1 - np.cos(np.pi * (end - t) / RAMP))
    return 1.0

def optical_envelope(t, start, end):
    if start < t < end:
        return OMEGA * np.sin(np.pi * (t - start) / (end - start))**2
    return 0.0

def flux(t):
    return PHI_PARK + (PHI_SWAP - PHI_PARK) * smooth_window(t, *TC)

def electrode_voltage(t):
    # E = V/d + E_stray; park voltage compensates the assumed static stray field.
    desired_field = E_PARK + (E_SWAP - E_PARK) * smooth_window(t, *AC)
    return GAP_CM * (desired_field - STRAY_FIELD)

def electric_field(t):
    return electrode_voltage(t) / GAP_CM + STRAY_FIELD

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
    return TWOPI * (transmon_frequency(flux(t)) - FC)

def delta_r0(t):
    return TWOPI * K_R0 * electric_field(t)**2

def delta_r1(t):
    return TWOPI * (F_A_ZERO - FC + K_R1 * electric_field(t)**2)

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

def results_table(ax, headers, rows, widths=None):
    """Render a shared numeric summary without plot axes."""
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                     cellLoc="center", bbox=(0, 0.20, 1, 0.58))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dc")
        if row == 0:
            cell.set_facecolor("#edf1f5")
            cell.set_text_props(weight="bold")
        elif row % 2:
            cell.set_facecolor("#fafbfc")
    return table


def basis_probabilities(mode, roundtrip=False, source="hyperfine", phase=None):
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
                    initial = retrieval_initial(alpha, beta, source)
                _, states, _, _, _ = evolve_read(initial, source=source)
                label = (("h0 → g", "h1 → e", "+ → +") if source == "hyperfine" else
                         ("r0 → g", "r1 → e", "+ → +"))[index]
                if roundtrip:
                    label += "\n" + ("After write/store" if stored else "Ideal memory")
            final = states[-1]
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
    return labels, np.asarray(rows), np.asarray(fidelities)


def plot_basis_columns(ax, labels, probabilities, title, mode="write"):
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
    ax.set(xlabel="Transmon input" if mode == "write" else "Memory input",
           ylabel="Joint output", zlabel="Probability", zlim=(0, 1), title=title)
    ax.tick_params(labelsize=7, pad=1)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 6
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.zaxis.label.set_size(9)
    ax.set_title(title, fontsize=10, pad=8)
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
    phase = calibrated_phase()
    U = memory_rotation(phase)

    # Nonzero input phase checks coherent transfer.
    theta = np.deg2rad(parameter("INPUT_THETA", 90.0))
    phi = np.deg2rad(parameter("INPUT_PHI", 60.0))
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    target = (alpha * h0 + beta * h1).unit()
    ts, states = simulate(alpha, beta, noisy=True, include_storage=not check)
    atomic_states = [rho.ptrace(2) for rho in states]
    fidelity = np.array([overlap(rho, target, U) for rho in atomic_states])
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
    print(f"Mapping overlap: {fidelity[mapping_index]:.6f}")
    print(f"Final overlap: {fidelity[-1]:.6f}")
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

    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    control_axis = fig.add_subplot(grid[0, 0])
    voltage_axis = control_axis.twinx()
    detuning_axis = fig.add_subplot(grid[0, 1])
    population_axis = fig.add_subplot(grid[1, :])
    wrong_axis = fig.add_subplot(grid[2, 0])
    basis_axis = fig.add_subplot(grid[2, 1], projection="3d")
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
    storage_duration = ts[-1] - ts[mapping_index]
    plot_preparation_probabilities(wrong_axis)
    labels, probabilities, input_fidelities = basis_probabilities("write", phase=phase)
    plot_basis_columns(basis_axis, labels, probabilities, "Write inputs (end of mapping)")
    fig.supxlabel(f"Selected input after {storage_duration:g} µs storage: "
                  f"phase-corrected overlap = {fidelity[-1]:.6f}; "
                  f"coherence = {coherence[-1]:.6f}.\n"
                  f"Phase-corrected joint fidelities: g={input_fidelities[0]:.4f}, "
                  f"e={input_fidelities[1]:.4f}, +={input_fidelities[2]:.4f}.\n"
                  "+ = equal coherent superposition; olive: intended logical outputs; Other: remaining joint states.",
                  fontsize=9)
    for ax in (control_axis, detuning_axis, population_axis):
        ax.set_xlabel("Time (us)")
        for start, end in (PREP, TC, AC, MAP):
            ax.axvline(start, color="gray", alpha=0.15)
            ax.axvline(end, color="gray", alpha=0.15)
    fig.suptitle("Transmon to atom")
    fig.canvas.manager.set_window_title("Transmon to atom")
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
        return PHI_PARK + (PHI_SWAP - PHI_PARK) * smooth_window(t, *tc)

    def voltage_read(t):
        field = E_PARK + (E_SWAP - E_PARK) * smooth_window(t, *ac)
        return GAP_CM * (field - STRAY_FIELD)

    def field_read(t):
        return voltage_read(t) / GAP_CM + STRAY_FIELD

    def dt_read(t):
        return TWOPI * (transmon_frequency(flux_read(t)) - FC)

    def d0_read(t):
        return TWOPI * K_R0 * field_read(t)**2

    def d1_read(t):
        return TWOPI * (F_A_ZERO - FC + K_R1 * field_read(t)**2)

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


def read_phase(source="hyperfine", roundtrip=False):
    # Fixed calibration from a nominal |+> input; never fit to noisy data
    # or separately to each unknown input state.
    if roundtrip:
        _, write_states = simulate(1, 1, noisy=False, include_storage=True)
        initial = write_states[-1]
    else:
        initial = retrieval_initial(1, 1, source)
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


def retrieval_main(source="hyperfine", roundtrip=False, check=False, save=False):
    if roundtrip and source != "hyperfine":
        raise ValueError("Round trip retrieves from hyperfine memory.")
    phase = read_phase(source, roundtrip)
    theta = np.deg2rad(parameter("INPUT_THETA", 90.0))
    phi = np.deg2rad(parameter("INPUT_PHI", 60.0))
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    target = (alpha * g + beta * e).unit()
    write_ts = write_states = None
    if roundtrip:
        # Continue the COMPLETE joint state. Do not reset, postselect,
        # or remove residual cavity photons/entanglement after storage.
        write_ts, write_states = simulate(alpha, beta, noisy=True, include_storage=True)
        initial = write_states[-1]
    else:
        initial = retrieval_initial(alpha, beta, source)
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
    print(f"Raw final transmon overlap: {raw[-1]:.6f}")
    print(f"Calibrated final transmon overlap: {corrected[-1]:.6f}")
    print(f"Joint overlap (target transmon, cavity vacuum, atom h0): {joint[-1]:.6f}")
    print(f"Final mean cavity photons: {qt.expect(n, states[-1]).real:.6e}")
    print(f"Final atomic h0 population: {qt.expect(Ph0, states[-1]).real:.6f}")
    print(f"Fixed calibrated transmon phase (rad): {phase:.6f}")
    print("Density-matrix checks passed.")

    if check:
        # Test a distinct relative phase with the SAME calibration.
        initial_test = retrieval_initial(1, 1j, source)
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
              "Ideal memory initialization", ts, states, corrected, joint)]
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    flux_axis = fig.add_subplot(grid[0, 0])
    voltage_axis = flux_axis.twinx()
    detuning_axis = fig.add_subplot(grid[0, 1])
    bottom_grid = grid[2, :].subgridspec(1, 2)
    wrong_axis = fig.add_subplot(bottom_grid[0, 0])
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
    plot_preparation_probabilities(wrong_axis)
    labels, probabilities, input_fidelities = basis_probabilities("read", roundtrip, source, phase=phase)
    for index, (label, _, _, _, _) in enumerate(cases):
        basis_axis = fig.add_subplot(bottom_grid[0, index + 1], projection="3d")
        selection = slice(3 * index, 3 * index + 3)
        title = "Retrieval: " + ("after write/store" if roundtrip else source)
        plot_basis_columns(basis_axis, labels[selection], probabilities[selection], title, mode="read")
        basis_axis.set_box_aspect((4, 4, 3), zoom=1.12)
    summary_lines = [f"{label}: selected-input overlap = {case_corrected[-1]:.6f}; "
                     f"joint overlap = {case_joint[-1]:.6f}"
                     for label, _, _, case_corrected, case_joint in cases]
    summary_lines.append("Fixed phase correction; basis columns use joint targets |g,0,h0⟩ / |e,0,h0⟩.")
    summary_lines.append(f"Phase-corrected joint fidelities: input 0={input_fidelities[0]:.4f}, "
                         f"input 1={input_fidelities[1]:.4f}, +={input_fidelities[2]:.4f}.")
    summary_lines.append("+ = equal coherent superposition; olive: intended logical outputs; Other: remaining joint states.")
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
    # Match population scales without assuming photon number is bounded by one.
    upper = max(ax.get_ylim()[1] for ax in population_axes)
    for ax in population_axes:
        ax.set_ylim(-0.03, max(1.03, upper))
    for ax in time_axes:
        ax.set(xlabel="Retrieval time (us)", xlim=(ts[0], ts[-1]))
        for _, (start, end) in stages:
            ax.axvline(start, color="gray", alpha=0.15)
            ax.axvline(end, color="gray", alpha=0.15)
    title = "Atom to transmon"
    subtitle = "After writing and storage" if roundtrip else f"Retrieval from {source} memory"
    fig.suptitle(title + "\n" + subtitle)
    fig.canvas.manager.set_window_title(title)
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
                label="Bare atom-cavity resonance")
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


def all_figures(save=False):
    """Build the two compact windows before a single blocking show call."""
    jobs = [
        ("Transmon to atom", lambda: main(save=save)),
        ("Atom to transmon", lambda: retrieval_main(roundtrip=True, save=save)),
    ]
    for index, (name, build) in enumerate(jobs, start=1):
        print(f"Generating figure {index}/{len(jobs)}: {name}", flush=True)
        build()
    count = len(plt.get_fignums())
    print(f"All {count} figures are ready. Close their windows to finish.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from hybrid_panel import launch
        launch(backend_path=Path(__file__).resolve())
        raise SystemExit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--leakage-scan", action="store_true")
    parser.add_argument("--retrieve", action="store_true",
                        help="Transfer a prepared atomic qubit to the transmon.")
    parser.add_argument("--roundtrip", action="store_true",
                        help="Write, store, then retrieve the same joint state.")
    parser.add_argument("--source", choices=("hyperfine", "rydberg"), default="hyperfine")
    parser.add_argument("--paper-figure", action="store_true", help="Generate a six-panel scientific summary.")
    parser.add_argument("--save", action="store_true",
                        help="Also export plot/data files; disabled by default.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open plot windows (for unattended runs).")
    parser.add_argument("--all-figures", action="store_true",
                        help="Generate the write/store and retrieval windows (default).")
    parser.add_argument("--write-paper-figure", action="store_true",
                        help="Paper-style preparation, dynamics, and write probabilities.")
    parser.add_argument("--run", action="store_true", help="Run figures directly, bypassing the startup panel.")
    parser.add_argument("--config", help="JSON object of parameter overrides.")
    parser.add_argument("--write-only", action="store_true", help="Only the write/store figure.")
    args = parser.parse_args()
    selected_modes = sum((args.paper_figure, args.retrieve, args.roundtrip,
                          args.leakage_scan, args.all_figures, args.write_paper_figure, args.write_only))
    if selected_modes > 1:
        parser.error("Choose one figure mode, or use --all-figures.")
    if args.all_figures and args.check:
        parser.error("--check runs verification separately from --all-figures.")
    if args.all_figures or (selected_modes == 0 and not args.check):
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

    if not args.no_show and not args.check:
        # Blocking interactive windows: close the windows to finish the run.
        plt.show(block=True)
    else:
        plt.close("all")
