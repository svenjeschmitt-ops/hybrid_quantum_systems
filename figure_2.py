import numpy as np
from scipy.constants import h, hbar, c, e
from scipy.linalg import expm
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Arc
from qutip_qip.circuit import QubitCircuit

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
        return P_u, N, n0, n1, u

    def figure_2a(self):

        P_u, N, n0, n1, u = self.projection()

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
        P_u, N, n0, n1, u = self.projection()
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
        P, N , n0, n1 , u=self.projection()
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

    def hamiltonian_r_error(self,tau_pi):
        b, r, r_til = [qt.basis(3, i) for i in range(3)]
        delta = 2 * np.pi * 1.32  # Fine-structure splitting, rad/ns

        # Equal optical couplings reproduce the published curves.
        coupling = r * b.dag() + r_til * b.dag()
        # H/hbar = (Omega/2) * (coupling + coupling.dag())
        pulse_area = (np.pi / 2) * (coupling + coupling.dag())
        generator = pulse_area + delta * tau_pi * r_til.proj()
        return generator

    def figure_2d(self):
        # Figure 2d sweeps pi-pulse length, with Omega * tau_pi = pi.
        # Each point is the population at the END of a different pulse.
        tau_pi = np.linspace(0, 3, 1001)  # ns
        b, r, r_til = [qt.basis(3, i) for i in range(3)]
        delta = 2 * np.pi * 1.32  # Fine-structure splitting, rad/ns

        # Equal optical couplings reproduce the published curves.
        coupling = r * b.dag() + r_til * b.dag()
        # H/hbar = (Omega/2) * (coupling + coupling.dag())
        #           + delta * |r_til><r_til|.

        pulse_area = (np.pi / 2) * (coupling + coupling.dag())
        populations = np.empty((len(tau_pi), 2))
        for i, duration in enumerate(tau_pi):
            generator = self.hamiltonian_r_error(duration)
            psi_final = (-1j * generator).expm() * b
            populations[i] = [
                qt.expect(r.proj(), psi_final),
                qt.expect(r_til.proj(), psi_final),
            ]

        ax = self.axs[1, 1]
        ax.plot(tau_pi, populations[:, 0], "b-", label=r"$43P_{1/2}$")
        ax.plot(tau_pi, populations[:, 1], "b--", label=r"$43P_{3/2}$")
        # Highlight the paper's operating point: tau_pi = 1 ns.
        psi_1ns = (-1j * (pulse_area + delta * r_til.proj())).expm() * b
        ax.plot(
            [1, 1],
            [qt.expect(r.proj(), psi_1ns), qt.expect(r_til.proj(), psi_1ns)],
            "ro", markerfacecolor="none",
        )
        ax.set(
            xlabel=r"$\tau_\pi^{(p)}$ (ns)", ylabel="Probability",
            xlim=(0, 3), ylim=(0, 1.02),
        )
        ax.legend(fontsize="small")
        return tau_pi, populations

    def figure_2f(self):
        qc =QubitCircuit(2,num_cbits=1,reverse_states=False)

        qc.add_gate("CNOT", controls=0, targets=1)
        qc.add_gate("SNOT",targets=0)

        qc.add_measurement("M0",targets=0,classical_store=0)
        qc.add_gate(
            "Z",
            targets=1,
            classical_controls=[0],
            classical_control_value=1,
        )
            
        # Custom circuit drawing.
        ax = self.axs[2, 1]
        ax.clear()

        top, bottom = 1.8, 0
        cx, hx, mx, zx = 1, 2, 3, 4.4
        width, height = 0.55, 0.55

        def line(xs, ys):
            ax.plot(xs, ys, color="black", lw=1.4, zorder=1)

        def gate(x, y, label=None):
            ax.add_patch(Rectangle(
                (x - width / 2, y - height / 2),
                width, height,
                facecolor="white", edgecolor="black",
                lw=1.4, zorder=3,
            ))
            if label:
                ax.text(
                    x, y, label, ha="center", va="center",
                    fontsize=13, zorder=4,
                )

        # Wires.
        line([0, mx - width / 2], [top, top])
        line([0, 5], [bottom, bottom])

        # CNOT.
        line([cx, cx], [bottom, top])
        ax.plot(cx, top, "ko", markersize=9)
        ax.add_patch(Circle(
            (cx, bottom), 0.18, fill=False,
            edgecolor="black", lw=1.4,
        ))
        line([cx - 0.18, cx + 0.18], [bottom, bottom])

        gate(hx, top, r"$H$")

        # Measurement symbol.
        gate(mx, top)
        ax.add_patch(Arc(
            (mx, top - 0.13), 0.44, 0.32,
            theta1=0, theta2=180, zorder=4,
        ))
        ax.annotate(
            "", xy=(mx + 0.12, top + 0.2),
            xytext=(mx - 0.06, top - 0.13),
            arrowprops=dict(arrowstyle="-|>", color="black"),
            zorder=4,
        )
        ax.text(mx, top - 0.5, r"$M$", ha="center")

        # Measurement result controls Z.
        for offset in (-0.045, 0.045):
            line(
                [mx + width / 2, zx + offset, zx + offset],
                [top + offset, top + offset, bottom + height / 2],
            )
        gate(zx, bottom, r"$Z^M$")

        # Stage boundaries.
        for i, x in enumerate([0.5, 1.5, 2.5, 3.7]):
            ax.axvline(
                x, color="steelblue", lw=1,
                linestyle=(0, (5, 4)), zorder=0,
            )
            ax.text(x + 0.08, -0.65, rf"$\Psi_{i}$", fontsize=11)

        # Input and output endpoints.
        for x, y, label in [
            (0, top, r"$\psi_a$"),
            (0, bottom, r"$|0\rangle$"),
            (5, bottom, r"$\phi_c$"),
        ]:
            ax.plot(
                x, y, "o", markerfacecolor="white",
                markeredgecolor="black", markersize=6, zorder=5,
            )
            ax.text(x, y - 0.4, label, ha="center", fontsize=12)

        ax.set(xlim=(-0.4, 5.5), ylim=(-0.8, 2.3))
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    def figure_2e(self):
        # Times in ns; Hamiltonians represent H/hbar in rad/ns.
        t1, t2, t3 = 1.0, 0.417, 1.0
        Omega = np.pi / t1
        delta_fs = 2 * np.pi * 1.32

        a, b, r, rt = [qt.basis(4, i) for i in range(4)]
        atoms = [a, b, r, rt]
        n0, n1 = [qt.basis(2, i) for i in range(2)]
        Ic = qt.qeye(2)
        Ia = qt.qeye(4)

        conversion = E_j / hbar * 1e-9

        def charge_blocks(ng):
            raw = self.H_c_calculation([ng])[0]

            # Remove intrinsic atomic energies in an atomic rotating frame.
            # Retain relative charging energies and Stark shifts.
            return [
                (
                    raw[i]
                    - (hbar * omega_u[i] / E_j) * Ic
                ) * conversion
                for i in range(3)
            ]

        def H_static(ng):
            Ha, Hb, Hr = charge_blocks(ng)

            # Approximation: same charge/Stark response for r and rt.
            Hrt = Hr + delta_fs * Ic

            return sum(
                qt.tensor(atom.proj(), block)
                for atom, block in zip(atoms, [Ha, Hb, Hr, Hrt])
            )

        H0 = H_static(0.0)
        H_switch = H_static(0.484)

        # Laser excitation operator in the 4 x 2 Hilbert space.
        D = qt.tensor(
            r * b.dag() + rt * b.dag(),
            Ic,
        )

        # Two optical tones resonant with the charge-dependent transitions.
        # These are residual frequencies in the chosen rotating frame.
        _, Hb, Hr = charge_blocks(0.0)
        detunings = np.array([
            qt.expect(Hr - Hb, n0),
            qt.expect(Hr - Hb, n1),
        ])

        def laser(t):
            return (Omega / 2) * np.sum(
                np.exp(-1j * detunings * t)
            )

        def laser_conj(t):
            return np.conj(laser(t))

        H_pulse = qt.QobjEvo([
            H0,
            [D, laser],
            [D.dag(), laser_conj],
        ])

        # Charge-qubit relaxation and pure dephasing.
        T1, T2 = 100.0, 10.0
        gamma_phi = 1 / T2 - 1 / (2 * T1)

        c_ops = [
            qt.tensor(Ia, np.sqrt(1 / T1) * n0 * n1.dag()),
            qt.tensor(Ia, np.sqrt(gamma_phi / 2) * qt.sigmaz()),
        ]

        inputs = [
            qt.tensor(a, n0),
            qt.tensor(a, n1),
            qt.tensor(b, n0),
            qt.tensor(b, n1),
        ]

        # Absolute times preserve the phases of the optical tones.
        stages = [
            (H_pulse,  [0.0, t1]),
            (H_switch, [t1, t1 + t2]),
            (H_pulse,  [t1 + t2, t1 + t2 + t3]),
        ]

        options = {
            "atol": 1e-9,
            "rtol": 1e-7,
            "max_step": 0.001,
            "nsteps": 100000,
        }

        probabilities = np.zeros((4, 4))
        end_states = []

        for i, psi in enumerate(inputs):
            rho = psi.proj()

            for H, times in stages:
                result = qt.mesolve(
                    H,
                    rho,
                    times,
                    c_ops=c_ops,
                    options=options,
                )
                rho = result.states[-1]

            end_states.append(rho)

            for j, output in enumerate(inputs):
                probabilities[i, j] = qt.expect(output.proj(), rho)

        # Replace the existing Figure 2e axes with a 3D subplot.
        self.fig.delaxes(self.axs[2, 0])
        ax = self.fig.add_subplot(3, 2, 5, projection="3d")
        self.axs[2, 0] = ax

        x, y = np.meshgrid(np.arange(4), np.arange(4))
        heights = probabilities.ravel()  # rows=input, columns=output

        ax.bar3d(
            x.ravel(),
            y.ravel(),
            np.zeros(16),
            0.65,
            0.65,
            heights,
            color="goldenrod",
            edgecolor="black",
            shade=True,
        )

        labels = [
            r"$|a,0\rangle$",
            r"$|a,1\rangle$",
            r"$|b,0\rangle$",
            r"$|b,1\rangle$",
        ]

        ax.set_xticks(np.arange(4) + 0.325)
        ax.set_yticks(np.arange(4) + 0.325)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Output")
        ax.set_ylabel("Input")
        ax.set_zlabel("Probability")
        ax.set_zlim(0, 1)


    def H_c_calculation(self, N_g0):
        P_u, N, n0, n1, u= self.projection()
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
test.figure_2d()
test.figure_2f()
test.figure_2e()
plt.show()       
