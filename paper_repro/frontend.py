from backend import quantumsystem
import numpy as np
from scipy.constants import h, hbar, c, e
import qutip as qt
import matplotlib.pyplot as plt



C_j = 30 * 10**(-18)
#E_j = 1.2 * 10**9
delta = 1
#delta_E = None
#P_u = None
n_hat = 2
N_g0 = np.linspace(0.45, 0.55, 1001)
f_ab = 2 * np.pi * 6.8 * 10**9
lamda = 297 * 10**(-9)
f_ar = c/lamda
alpa_u = np.array([0.079, 0.079, 98.3e6]) * h / 100**2
V_g = 0.603229e-3

#Q = quantumsystem()

#defining hilberspace
a = qt.basis(3,0)
b = qt.basis(3,1)
r = qt.basis(3,2)
u = [a,b,r]

n0 = qt.basis(2,0)
n1 = qt.basis(2,1)
n_hat = [n0,n1]
N = n1 * n1.dag()
for i in range(len(u)):
    for j in range(len(n_hat)):
        hilberspace = qt.tensor(u[i],n_hat[j])

P_a = a * a.dag()
P_b = b * b.dag()
P_c = r * r.dag()

P_u = [P_a,P_b,P_c]

E_a = 0 
E_b = (f_ab/2* np.pi) * h
E_r = 2 * np.pi * f_ar

E_u = [E_a, E_b, E_r]

omega_a = 0
omega_b = f_ab
omega_r = 2 * np.pi * f_ar

omega_u = [omega_a, omega_b, omega_r]
C_g0 = 265.6e-18
eps = 1
s = 6e-6 * 6e-6
l = 1.2e-6
V = s *l
C_g = np.zeros(len(alpa_u))
for i in range(len(alpa_u)):
    C_g[i] = C_g0 + (s/l) * (alpa_u[i] / V)
#Constance
E_c = (2*e)**2 / (2*(C_g[0]+C_j))
N_g = (C_g[0] * V_g)/2*e
E_j = h * 1.2e9
H_t = []

tunnelterm = n0* n1.dag()+n1 * n0.dag()

#calculation for total hamiltonian
H_c = []
#print(N_g0)
for i in range(len(N_g0)):
    H_c.append((E_c/E_j) * (N - N_g0[i] * qt.qeye(2))**2 - 1/2 * tunnelterm) # total len 1001
    #print(f"H_c: {H_c[i]}")
delta_E = []
for i in range(len(E_u)):
    delta_E.append(-1/2 * alpa_u[0] * E_u[i]**2) # total len 3
#2x2 matrix
H_c_pl_term_1 = [[] for _ in range(len(H_c))]
for j in range(len(H_c)):
    for i in range(3):
        H_c_pl_term_1[j].append(H_c[j] + (hbar * omega_u[i] + delta_E[i]) * qt.qeye(2))
    #print(f"term1: {H_c_pl_term_1}")
print(np.array(H_c_pl_term_1).shape)
H_t = [[] for _ in range(len(H_c))]
#Total hamiltonian; 
for j in range(len(H_c_pl_term_1)):
    for i in range(len(omega_u)):
        H_t[j].append(qt.tensor(P_u[i],H_c_pl_term_1[j][i]))

energies = [[] for _ in range(len(H_t))]

#Einstates calculation
for j in range(len(H_t)):
    for i in range(len(omega_u)):
        energies[j].append(qt.Qobj(H_t[j][i]).eigenenergies())
print(f" energies: {energies[1000][0]}")
print(f"shape of energies: {np.array(energies).shape} ")
#figure 1 plot a
f1_p_a_y = [[] for _ in range(len(H_t))]
for j in range(len(H_t)):
    for i in range(3):
        f1_p_a_y[j].append(energies[j][i] - hbar * omega_u[i] / E_j)
print(np.array(f1_p_a_y).shape)
#print(f1_p_a_x)
f1_p_a_x = N_g0
fig, axs = plt.subplots(3,2)
y_a = np.array([row[0] for row in f1_p_a_y])

axs[0,0].plot(f1_p_a_x,y_a)
axs[0,0].set_ylim(45,65)
#axs[0,0].set_ylabel("\u03B5 -  \hbar\u03A9 / E_j" )
plt.show()
