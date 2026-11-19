"""Effective optical pulses with independently calibrated transition strengths."""
import numpy as np


def optical_detuning(config, arm, field):
    """Laser minus atomic transition (MHz), at the nominal laser-on field."""
    transition = config[f'OPTICAL_TRANSITION_{arm}']
    mapping = config[f'OPTICAL_MAPPING_{arm}']
    if transition is None:
        return 0.0
    stark = config['K_R0' if arm == 0 else 'K_R1']
    return (mapping-transition) - stark*field**2


def build_optics(b, pulses, nominal_field):
    """Build drives with carrier offsets relative to each nominal transition."""
    import qutip as qt
    h = qt.QobjEvo(0*b.Pr0)
    for arm, interval in pulses:
        reference = b.TWOPI*(b.atomic_energy_MHz(f'r{arm}', nominal_field, offsets=False)
                              + optical_detuning(b._SETTINGS, arm, nominal_field))
        transition = b.Bprep if arm == 0 else b.Bmap1.dag()
        def coefficient(t, window=interval, carrier=reference, index=arm):
            return .5*b.optical_envelope(t, *window, arm=index)*np.exp(-1j*carrier*(t-window[0]))
        drive = qt.QobjEvo([transition, coefficient])
        h += drive+drive.dag()
    return h, []
