"""Four-level atom, multilevel transmon, and cavity modes in the RWA.

All Hamiltonians are expressed as H/hbar in consistent angular-frequency
and time units. Tensor order is cavity modes, transmon, then atom.
"""

from numbers import Integral

import numpy as np
import qutip as qt

ATOM_A, ATOM_B, ATOM_R, ATOM_R_TIL = range(4)


def _real_vector(values, name):
    values = np.asarray(values, dtype=complex)
    if (values.ndim != 1 or not np.all(np.isfinite(values))
            or np.any(values.imag != 0)):
        raise ValueError(f"{name} must be a vector of finite real values.")
    return values.real


def _index(value, size, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral) or not 0 <= value < size):
        raise ValueError(f"{name} must be an integer in [0, {size}).")
    return int(value)


def _coupling(value, name):
    value = complex(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def make_rydberg_laser(omega_L, Omega_r, Omega_r_til, initial_state=ATOM_B):
    """Return desired and leakage transitions driven by the same laser.

    omega_L is the laser angular frequency. Omega_r(t) and Omega_r_til(t)
    are callable complex Rabi envelopes. Each raising coefficient is
    Omega(t)/2 * exp(-1j*omega_L*t), in the laboratory frame.
    """
    initial_state = _index(initial_state, 4, "initial_state")
    if initial_state not in (ATOM_A, ATOM_B):
        raise ValueError("initial_state must be ATOM_A or ATOM_B.")
    omega_L = _real_vector([omega_L], "omega_L")[0]
    if not callable(Omega_r) or not callable(Omega_r_til):
        raise TypeError("Omega_r and Omega_r_til must be functions of time.")

    def desired_coefficient(t):
        return 0.5 * Omega_r(t) * np.exp(-1j * omega_L * t)

    def leakage_coefficient(t):
        return 0.5 * Omega_r_til(t) * np.exp(-1j * omega_L * t)

    return [
        (initial_state, ATOM_R, desired_coefficient),
        (initial_state, ATOM_R_TIL, leakage_coefficient),
    ]


def build_hamiltonian(
    cutoffs, wc, wt, wa, gt,
    atom_couplings=(), cavity_drives=(), laser_drives=(),
):
    """Build the entire modeled system's H(t)/hbar as a QuTiP QobjEvo.

    cutoffs: Fock dimensions (integers >= 2), one per cavity mode.
    wc: angular frequency of each cavity mode.
    wt: transmon level energies/hbar, retaining at least two levels.
    wa: atomic energies/hbar in the order a, b, r, r_til.
    gt: complex array (number of modes, number of transmon levels - 1).
        gt[m, j] multiplies a_m.dag() * |j><j+1|. Include the charge
        matrix element in gt; no additional sqrt(j+1) is applied.
    atom_couplings: (mode, lower, upper, g) entries multiplying
        a_mode.dag() * |lower><upper|. Supply physically allowed
        transitions, with wa[upper] > wa[lower].
    cavity_drives: (mode, f) entries; callable f(t) multiplies a_mode.dag().
    laser_drives: (initial, final, f) entries; callable f(t) multiplies
        |final><initial|. Include the Rabi factor 1/2 and carrier phase.

    Each coupling/drive automatically includes its Hermitian conjugate.
    Tensor order is cavity mode 0, ..., transmon, atom. Free energies and
    drive phases must use the same frame (laboratory frame for the helper
    make_rydberg_laser). Use consistent angular-frequency and time units;
    pass the result directly to a QuTiP solver without multiplying by hbar.
    Loss and decay belong in separate collapse operators.
    """
    cutoffs = list(cutoffs)
    M = len(cutoffs)
    if M == 0:
        raise ValueError("At least one cavity mode is required.")
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, Integral)
           or d < 2 for d in cutoffs):
        raise ValueError("cutoffs must contain integer dimensions >= 2.")
    cutoffs = [int(d) for d in cutoffs]
    wc = _real_vector(wc, "wc")
    wt = _real_vector(wt, "wt")
    wa = _real_vector(wa, "wa")
    Nt, Na = len(wt), len(wa)
    if len(wc) != M:
        raise ValueError("wc must contain one frequency per cavity mode.")
    if Nt < 2:
        raise ValueError("Retain at least two transmon levels.")
    if Na != 4:
        raise ValueError("wa must contain energies for a, b, r, and r_til.")
    gt = np.asarray(gt, dtype=complex)
    if gt.shape != (M, Nt - 1) or not np.all(np.isfinite(gt)):
        raise ValueError(f"gt must be finite and have shape {(M, Nt - 1)}.")

    dims = cutoffs + [Nt, Na]
    identities = [qt.qeye(d) for d in dims]

    def embed(op, slot):
        factors = identities.copy()
        factors[slot] = op
        return qt.tensor(*factors)

    cavity_a = [embed(qt.destroy(d), m) for m, d in enumerate(cutoffs)]
    transmon_states = [qt.basis(Nt, j) for j in range(Nt)]
    atom_states = [qt.basis(Na, u) for u in range(Na)]
    H_static = 0 * cavity_a[0]

    # Free energies of every retained subsystem.
    for mode, a_mode in enumerate(cavity_a):
        H_static += wc[mode] * a_mode.dag() * a_mode
    for j, state in enumerate(transmon_states):
        H_static += wt[j] * embed(state.proj(), M)
    for u, state in enumerate(atom_states):
        H_static += wa[u] * embed(state.proj(), M + 1)

    # Transmon-cavity exchange.
    for j in range(Nt - 1):
        lower = embed(transmon_states[j] * transmon_states[j + 1].dag(), M)
        for mode in range(M):
            interaction = gt[mode, j] * cavity_a[mode].dag() * lower
            H_static += interaction + interaction.dag()

    # Selected atom-cavity exchange; atomic indices need not follow energy order.
    for mode, lo, hi, g in atom_couplings:
        mode = _index(mode, M, "atom coupling mode")
        lo = _index(lo, Na, "lower atomic index")
        hi = _index(hi, Na, "upper atomic index")
        if wa[hi] <= wa[lo]:
            raise ValueError("Atom-cavity upper state must have higher energy.")
        lower = embed(atom_states[lo] * atom_states[hi].dag(), M + 1)
        interaction = _coupling(g, "g") * cavity_a[mode].dag() * lower
        H_static += interaction + interaction.dag()

    terms = [H_static]

    def add_drive(raising_operator, f):
        if not callable(f):
            raise TypeError("Drive coefficients must be functions of time.")

        def conjugate_coefficient(t):
            return np.conj(f(t))

        terms.append([raising_operator, f])
        terms.append([raising_operator.dag(), conjugate_coefficient])

    for mode, f in cavity_drives:
        mode = _index(mode, M, "cavity drive mode")
        add_drive(cavity_a[mode].dag(), f)

    for initial, final, f in laser_drives:
        initial = _index(initial, Na, "initial atomic index")
        final = _index(final, Na, "final atomic index")
        if initial == final:
            raise ValueError("Laser transitions must connect distinct states.")
        raising = embed(atom_states[final] * atom_states[initial].dag(), M + 1)
        add_drive(raising, f)

    return qt.QobjEvo(terms)


class quantumsystem:
    """Compatibility wrapper exposing the Hamiltonian builders on an instance."""

    hamiltonian = staticmethod(build_hamiltonian)
    build_hamiltonian = staticmethod(build_hamiltonian)
    make_rydberg_laser = staticmethod(make_rydberg_laser)
