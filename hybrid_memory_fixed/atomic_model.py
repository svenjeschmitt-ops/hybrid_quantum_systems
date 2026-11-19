"""Quadratic atomic Stark energies and constant cavity couplings."""
import math
import numpy as np

DIPOLE_TO_MHZ = 1.602176634e-19 * 5.29177210544e-11 / 6.62607015e-34 / 1e6


def atomic_levels(config):
    return {'r0': (0, 0., config['K_R0']),
            'r1': (1, config['F_A_ZERO'], config['K_R1']),
            'wrong': (0, config['F_WRONG_SPLIT'], config['K_WRONG'])}


def cavity_links(config):
    return {'r1<-r0': 1.}


class AtomicModel:
    def __init__(self, config):
        self.config = config
        self.levels, self.links = atomic_levels(config), cavity_links(config)

    def energy(self, name, field, derivative=False):
        field = np.asarray(field)
        _, offset, stark = self.levels[name]
        return 2*stark*field if derivative else offset+stark*field**2

    def transition(self, field):
        return self.energy('r1', field)-self.energy('r0', field)

    def coupling(self, link, field, derivative=False):
        field = np.asarray(field)
        if derivative:
            return np.zeros_like(field, dtype=float)
        c = self.config
        base = (c['G_AC_MHZ'] if c['DIRECT_COUPLING'] else
                DIPOLE_TO_MHZ*c['D_01_EA0']*c['E_ZPF_V_PER_M']*abs(c['DIPOLE_ALIGNMENT']))
        return base*self.links[link]+np.zeros_like(field, dtype=float)

    def resonance(self, reference):
        stark = self.config['K_R1']-self.config['K_R0']
        if stark == 0:
            if not math.isclose(self.config['F_A_ZERO'], self.config['FC'], abs_tol=1e-9, rel_tol=0):
                raise ValueError('Equal Stark coefficients cannot tune the atom to cavity resonance.')
            return 0.
        square = (self.config['FC']-self.config['F_A_ZERO'])/stark
        if square < 0:
            raise ValueError('The atomic Stark model cannot reach cavity resonance.')
        return math.sqrt(square)
