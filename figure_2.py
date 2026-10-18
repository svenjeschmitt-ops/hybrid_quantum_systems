import numpy as np
from scipy.constants import h, hbar, c, e
import qutip as qt
import matplotlib.pyplot as plt

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
class figure_2:
    def __init__(self):
        self.fig,self.axs = plt.subplots(3,2,constrained_layout=True)
            
    def projection(self):
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
        C_g = np.zeros(len(alpa_u))
        for i in range(len(alpa_u)):
            C_g[i] = C_g0 + (s/l) * (alpa_u[i] / V)
        #Constance
        E_c = (2*e)**2 / (2*(C_g[0]+C_j))
        H_t = []

        tunnelterm = n0 * n1.dag()+n1 * n0.dag()

        #calculation for total hamiltonian
        H_c = []
        for i in range(len(N_g0)):
            H_c.append((E_c/E_j) * (N - N_g0[i] * qt.qeye(2))**2 - 1/2 * tunnelterm) # Hamiltonian of the cooper pair box

        delta_E = []
        for i in range(len(E_u)):
            delta_E.append(-1/2 * alpa_u[0] * E_u[i]**2) #Energy shift 

        H_c_pl_term_1 = [[] for _ in range(len(H_c))]
        H_t = [[] for _ in range(len(H_c))]
        for j in range(len(H_c)):
            for i in range(3):
                H_c_pl_term_1[j].append(H_c[j] + (hbar * omega_u[i] + delta_E[i]) * qt.qeye(2))
                H_t[j].append(qt.tensor(P_u[i],H_c_pl_term_1[j][i]))
        self.H_c_pl_term_1 = H_c_pl_term_1
        sector_energies, states = self.energies_states(H_c_pl_term_1)
        #figure 1 plot a
        f1_p_a_y = [[] for _ in range(len(H_t))]
        for j in range(len(H_t)):
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



test = figure_2()

test.figure_2a()
test.figure_2b()
plt.show()       
