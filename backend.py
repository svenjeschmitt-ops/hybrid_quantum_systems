import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar,pi


#adjustable valies: N, omega, E_j 
E_j = np.zeros(N) 
omega_u = np.zeros(len(lamda)) 
N_atom
N_trans
E_y
g_cav_trans
coup_trans_cav #
coup_atom_cav



#field parameter:
f
amp
t
phase

# microwave cavity field 
omega


class quantumsystem:
    def __init__(self) -> qt.Qobj:
        return qt.Qobj


    def hamiltonian(self, N_transmon):

        def H_transmon(transmon_state):
            trasmon_state = [qt.basis(N_trans,i) for i in range(N_transmon)]

            H_t = []
            for i in range(len(trasmon_state)):
                H_t.append(E_j[i] * trasmon_state[i].proj()) # transmon hamiltonian
            return H_t

        def H_cavity(N_cavity):
            a = qt.create(N_cavity)
            a_pl = qt.destroy(N_cavity)
            H_c = []
            for i in range(len(omega_u)):
                H_c.append(hbar * omega_u[i] * a_p * a) # cavity hamilitonian 
            return H_c

        def H_atom(N_atom):
            atom_state = [qt.basis(N_atom,i)]
            H_a = []
            for i in range(len(atom_state)):
                H_a = E_y * atom_state[i].proj() # atom hamiltonian
            return H_a

        def H_transmon_cavity_interface(atom_state,omega_u, trasmon_state):
            H_tc = []
            H_tc_c =[]
            for j in range(len(atom_state)):
                for i in range(len(omega_u)):
                    H_tc_c.append(hbar * (a+a_pl) * len(omega_u) * coup_trans_cav[i][j] * trasmon_state[j] * trasmon_state[j+1].dag())
                H_tc.append(H_tc_c) # transmon-cavity interface hamiltonian
            return H_tc

        def H_atom_cavity_interface(transmon_state, omega_u, atom_state):
            H_ac = []
            H_ac_c = []
            for i in range(len(trasmon_state)):
                for i in range(len(omega_u)):
                    H_ac_c.append(hbar * (a+a_pl)*len(omega_u) * coup_atom_cav[i][j] * atom_state[j] * atom_state[j+1].dag())
                H_ac.append(H_ac_c) # atom-cavity interface hamiltonian
            return H_ac
        
        def H_microwave_cavity_field(f,amp, t,phase):
            H_mw = []
            H_mw_c = []
            def epsi(f,amp, t,phase):
                return amp * np.exp(1j * phase) * np.exp(2*pi*t*(1/f))
            for i,val in enumerate(omega_u):
                H_mw.append(hbar*epsi(f,amp,t,phase)* qt.Qobj((-1j * omega_u * t)).expm() * a_pl + np.conj(epsi(f,amp,t,phase)) * qt.Qobj((-1j * omega_u * t)).expm() * a)
            return H_mw

        def H_laser_excitation(Omega,t,delta):
            b,r,r_til = [qt.basis(3,i) for i in range(3)]
            states = r * b.dag() + r_til * b.dag()
            H_l = (Omega/2) * (np.exp(-1j * delta * t) * (states + states.dag()))
            return H_l

        H_t = H_transmon()
        H_c = H_cavity()
        H_a = H_atom()
        H_tc = H_transmon_cavity_interface()
        H_ac = H_atom_cavity_interface()
        H_l = H_laser_excitation()
        H_m = H_microwave_cavity_field()

        H_total = H_t + H_c + H_a + H_tc + H_ac + H_l + H_m 
        return H_com

    def eigenstate_calculation(H):
        
        return eng, state
    
    def in_states():
        return

    def trasfer_protocol():
        return 

    def solve_schroedinger_eq():
        return

    def fidi_calculation(): 
        return 
