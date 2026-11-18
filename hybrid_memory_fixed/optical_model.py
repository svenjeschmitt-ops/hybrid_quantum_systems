"""Effective optical pulses with independently calibrated transition strengths."""
import numpy as np


def build_optics(b, pulses, nominal_field):
    """Build both resonant drives; each arm retains its own envelope and carrier."""
    import qutip as qt
    h = qt.QobjEvo(0*b.Pr0)
    for arm, interval in pulses:
        reference = b.TWOPI*b.atomic_energy_MHz(f'r{arm}', nominal_field, offsets=False)
        transition = b.Bprep if arm == 0 else b.Bmap1.dag()
        def coefficient(t, window=interval, carrier=reference, index=arm):
            return .5*b.optical_envelope(t, *window, arm=index)*np.exp(-1j*carrier*(t-window[0]))
        drive = qt.QobjEvo([transition, coefficient])
        h += drive+drive.dag()
    return h, []
