import numpy as np
from scipy.constants import h, hbar, c, e
from scipy.linalg import expm
import qutip as qt
import matplotlib.pyplot as plt
I = qt.qeye(2)
C_j = 30 * 10**(-18)
delta = 1
N_g0 = np.linspace(0.45, 0.55, 1001)
f_ab = 2 * np.pi * 6.8 * 10**9
lamda = 297 * 10**(-9)
f_ar = c/lamda
alpa_u = np.array([0.079, 0.079, 98.3e6]) * h / 100**2
V_g = 0.603229e-3


omega_a = 0
omega_b = f_ab
omega_r = 2 * np.pi * f_ar

omega_u = [omega_a, omega_b, omega_r]

C_g0 = 265.6e-18
eps = 1
s = 6e-6 * 6e-6
l = 1.2e-6
V = s *l

E_j = h * 1.2e9
E_a = 0 
E_b = (f_ab/2* np.pi) * h
E_r = 2 * np.pi * f_ar
E_u = [E_a, E_b, E_r]

C_g = np.zeros(len(alpa_u))
for i in range(len(alpa_u)):
    C_g[i] = C_g0 + (s/l) * (alpa_u[i] / V)
#Constance
E_c = (2*e)**2 / (2*(C_g[0]+C_j))
class figure_2:
    def __init__(self):
        self.fig,self.axs = plt.subplots(3,2,constrained_layout=True)
            
    def projection(self): #giving back P_u, N, n0, n1
        a = qt.basis(3,0)
        b = qt.basis(3,1)
        r = qt.basis(3,2)
        u = [a,b,r]
        n0 = qt.basis(2,0)
        n1 = qt.basis(2,1)
        N = n1 * n1.dag()

        P_a = a * a.dag()
        P_b = b * b.dag()
        P_c = r * r.dag()

        P_u = [P_a,P_b,P_c]
        return P_u, N, n0, n1

    def figure_2a(self):

        P_u, N, n0, n1 = self.projection()

        self.tunnelterm = n0 * n1.dag()+n1 * n0.dag()

        #calculation for total hamiltonian
        self.H_c_pl_term_1 = self.H_c_calculation(N_g0)
        
        H_c_pl_term_1 = self.H_c_pl_term_1
        
        sector_energies, states = self.energies_states(H_c_pl_term_1)
        #figure 1 plot a
        f1_p_a_y = [[] for _ in range(len(sector_energies))]
        for j in range(len(sector_energies)):
            for i in range(3):
                # Diagonalize each independent block to retain the a, b, r labels.
                f1_p_a_y[j].append(sector_energies[j][i] - hbar * omega_u[i] / E_j)
        f1_p_a_x = N_g0
        
        y_a = np.array([row[0] for row in f1_p_a_y])
        self.plot_set((0,0),f1_p_a_x,y_a, (45,65))

        
    def plot_set(self,pos:tuple, x, y,ylim=None):
        ax = self.axs[pos[0],pos[1]]
        ax.plot(x,y)
        if ylim is not None:
            ax.set_ylim(*ylim)


    def figure_2b(self):
        energies, states = self.energies_states(self.H_c_pl_term_1)
        P_u, N, n0, n1 = self.projection()
        except_val =[]
        for row in states:
            # Each sector contains a flat array of Qobj eigenstates.
            except_val.append([
                [qt.expect(N,state) for state in both_state]
                for both_state in row])
        #print(except_val[:,0,0])
        except_val = np.asarray(except_val)
        print(except_val[:,0])
        self.plot_set((0,1),N_g0,except_val[:,0])

    def figure_2c(self):
        P, N , n0, n1 =self.projection()
        t = np.linspace(0, 1.5, 1001)

        N_g0_del = 0.016
        N_g0_ab = 0.5
        N_g0_r = N_g0_ab - N_g0_del

        hamiltonians = self.H_c_calculation([N_g0_r])[0]
        # Work in ns: Hamiltonians passed to mesolve have units rad/ns.
        # Table I: population relaxation T1 and coherence decay T2.
        T1, T2 = 100.0, 10.0
        gamma_phi = 1 / T2 - 1 / (2 * T1)
        collapse_ops = [
            np.sqrt(1 / T1) * n0 * n1.dag(),
            np.sqrt(gamma_phi / 2) * (qt.qeye(2) - 2 * N),
        ]
        ax = self.axs[1, 0]
        for sector, label, style in [
            (0, r"$|a,0\rangle\to|a,1\rangle$", "k-"),
            (1, r"$|b,0\rangle\to|b,1\rangle$", "k:"),
            (2, r"$|r,0\rangle\to|r,1\rangle$", "b--"),
        ]:
            H = hamiltonians[sector]
            # Scalar energies only add a global phase; remove them for accuracy.
            H = (H - (H.tr() / 2) * qt.qeye(2)) * E_j / hbar * 1e-9
            result = qt.mesolve(
                H, n0, t, c_ops=collapse_ops, e_ops=[N],
                options={"atol": 1e-10, "rtol": 1e-8},
            )
            ax.plot(t, result.expect[0], style, label=label)
        ax.set(xlabel="Time (ns)", ylabel="Probability", xlim=(0, 1.5), ylim=(0, 1))
        ax.legend(fontsize="small")

    def energies_states(self, ham: list) -> tuple[list, list]:

        sector_energies = []
        states = []

        for row in ham:
            sta = []
            ener = []

            for i in row:
                # Diagonalize each independent block to retain the a, b, r labels.
                energy, stat = i.eigenstates()
                sta.append(stat)
                ener.append(energy)
            sector_energies.append(ener)
            states.append(sta)
            
        return sector_energies, states
    def H_c_calculation(self, N_g0):
        P_u, N, n0, n1 = self.projection()
        identity = qt.qeye(2)
        tunnelterm = n0 * n1.dag() + n1 * n0.dag()
        C_total0 = C_g0 + C_j
        E_c0 = (2 * e)**2 / (2 * C_total0)
        field0 = 2 * e / (l * C_total0)  # V/m
        eta = (C_g - C_g0) / C_total0

        # Each row is one empty gate bias; columns are atomic sectors a, b, r.
        # Return H/E_j consistently, including the atomic and Stark energies.
        hamiltonians = []
        for ng0 in N_g0:
            voltage = 2 * e * ng0 / C_g0
            nj = C_j * voltage / (2 * e)
            row = []
            for u in range(3):
                ec_u = E_c0 / (1 + eta[u])
                ng_u = ng0 + eta[u] * (ng0 + nj)
                field_u = field0 * (N + nj * identity) / (1 + eta[u])
                # Normalize coefficients before Qobj arithmetic so tiny joule
                # entries are not discarded by sparse-matrix cleanup.
                stark_u = (-0.5 * alpa_u[u] / E_j) * field_u**2
                H_charge = (ec_u / E_j) * (N - ng_u * identity)**2 - 0.5 * tunnelterm
                row.append(H_charge + stark_u + (hbar * omega_u[u] / E_j) * identity)
            hamiltonians.append(row)
        return hamiltonians


test = figure_2()

test.figure_2a()
test.figure_2b()
test.figure_2c()
plt.show()       
