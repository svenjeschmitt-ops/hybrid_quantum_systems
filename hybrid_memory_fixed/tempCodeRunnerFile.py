"""Standard lifetime and pure-dephasing collapse operators."""

def collapse_operators(b, flux, field, optical0, optical1):
    """Construct the standard lifetime and pure-dephasing channels."""
    import numpy as np
    ops = [np.sqrt(1/b.T1_T)*b.sm, np.sqrt(1/b.T1_C)*b.a]
    lifetimes = dict(r0=b.TAU_R0, r1=b.TAU_R1, wrong=b.TAU_WRONG,
                     **{x['label']: x['lifetime_us'] for x in b.EXTRA_LEVELS})
    for source, lifetime in lifetimes.items():
        target = 'r0' if source == 'r1' else 'lost'
        ops.append(np.sqrt(1/lifetime)*b.A(b.ATOM_STATES[target], b.ATOM_STATES[source]))
    ops.extend([np.sqrt(1/(2*b.TPHI_T))*b.Zt,
                np.sqrt(1/(2*b.TPHI_R))*b.Zr,
                np.sqrt(1/(2*b.TPHI_H))*b.Zh])
    return ops


def snapshot(operators, time):
    """Freeze time-dependent channels at parked controls for constant propagation."""
    import qutip as qt
    return [op(time) if isinstance(op, qt.QobjEvo) else op for op in operators]
